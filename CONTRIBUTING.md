# 参与开发

1. 从默认分支拉一个功能分支。
2. 不要提交 `.venv/`、`desktop/target/`、`dist/`、模型文件或密钥。
3. 改动识别或界面后，运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile sysaudio_stt/*.py
```

4. 提交说明用 Conventional Commits，例如 `fix: 流式断句后保留标点`。
5. 在 GitLab 上开 Merge Request，写清改了什么、怎么验证。
