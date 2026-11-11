use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static ENGINE: Mutex<Option<Child>> = Mutex::new(None);

fn project_root() -> PathBuf {
    if let Ok(root) = std::env::var("SYSAUDIO_ROOT") {
        return PathBuf::from(root);
    }
    let mut dir = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if dir.join("sysaudio_stt").join("__main__.py").is_file() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn python_bin(root: &std::path::Path) -> PathBuf {
    if cfg!(windows) {
        let win = root.join(".venv").join("Scripts").join("python.exe");
        if win.is_file() {
            return win;
        }
        return PathBuf::from("python");
    }
    let mac = root.join(".venv").join("bin").join("python");
    if mac.is_file() {
        return mac;
    }
    PathBuf::from("python3")
}

fn start_engine() -> String {
    let root = project_root();
    let python = python_bin(&root);
    let mut child = Command::new(python)
        .arg("-m")
        .arg("sysaudio_stt")
        .arg("--serve")
        .current_dir(&root)
        .env("PYTHONPATH", &root)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("无法启动识别引擎");
    let stdout = child.stdout.take().expect("引擎没有输出");
    let reader = BufReader::new(stdout);
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut lines = reader.lines();
    let mut url = String::new();
    while Instant::now() <= deadline {
        match lines.next() {
            Some(Ok(line)) => {
                eprintln!("{line}");
                if let Some(rest) = line.strip_prefix("READY ") {
                    url = rest.trim().to_string();
                    break;
                }
            }
            _ => break,
        }
    }
    std::thread::spawn(move || {
        for line in lines {
            if let Ok(line) = line {
                eprintln!("{line}");
            }
        }
    });
    if url.is_empty() {
        let _ = child.kill();
        panic!("识别引擎没有就绪");
    }
    *ENGINE.lock().expect("engine lock") = Some(child);
    url
}

fn stop_engine() {
    if let Some(mut child) = ENGINE.lock().expect("engine lock").take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn main() {
    let url = start_engine();
    let parsed = url.parse().expect("地址无效");
    tauri::Builder::default()
        .setup(move |app| {
            let icon = tauri::image::Image::from_bytes(include_bytes!("../icons/icon.png"))?;
            tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::External(parsed))
                .title("VoiceCat")
                .inner_size(1100.0, 760.0)
                .min_inner_size(860.0, 560.0)
                .icon(icon)?
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("无法创建桌面窗口")
        .run(|_app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                stop_engine();
            }
        });
}
