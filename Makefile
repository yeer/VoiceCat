.PHONY: macos windows
macos:
	cd desktop && cargo build
windows:
	@echo Windows 包必须在 Windows 上执行: cd desktop && cargo build
	@echo 系统声音走 WASAPI 环回，需要 pip install pyaudiowpatch
