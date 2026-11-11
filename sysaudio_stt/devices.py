"""Routing hints. Never selects a microphone as a silent fallback."""

from __future__ import annotations


def _name(device: dict) -> str:
    return str(device.get("name") or "")


def _is_blackhole(device: dict) -> bool:
    return "blackhole" in _name(device).lower()


def blackhole_routed_name(devices: list[dict]) -> str | None:
    """Return a BlackHole device only when system output is actually routed into it."""
    blackholes = [device for device in devices if _is_blackhole(device)]
    if not blackholes:
        return None
    outputs = [
        device
        for device in devices
        if device.get("is_default_system_output") or device.get("is_default_output")
    ]
    for output in outputs:
        if _is_blackhole(output):
            return _name(output)
        sub_uids = set(output.get("subdevice_uids") or [])
        for hole in blackholes:
            if hole.get("uid") and hole.get("uid") in sub_uids:
                return _name(hole)
    return None


def routing_advice(devices: list[dict]) -> list[str]:
    by_uid = {device.get("uid"): _name(device) for device in devices}
    system = next((device for device in devices if device.get("is_default_system_output")), None)
    lines = ["建议: 优先用 --source system 捕获系统输出，不依赖虚拟声卡，也不会改默认设备。"]
    if system is None:
        lines.append("没有读到默认系统输出设备。")
        return lines
    subs = [by_uid.get(uid, uid) for uid in (system.get("subdevice_uids") or [])]
    if subs:
        lines.append("当前系统输出「%s」的子设备: %s" % (_name(system), "、".join(subs)))
    routed = blackhole_routed_name(devices)
    if routed:
        lines.append('BlackHole 已在系统输出路径中，也可以用 --source device --device "%s"。' % routed)
    elif any(_is_blackhole(device) for device in devices):
        lines.append("检测到 BlackHole，但它不在当前系统输出里。直接采集它可能听不到正在播放的声音。")
        lines.append("若要走虚拟声卡：在「音频 MIDI 设置」里把扬声器和 BlackHole 放进同一个多输出设备，并把该设备设为系统输出。")
    else:
        lines.append("未检测到 BlackHole。系统 tap 或 --source sckit 仍可捕获系统输出。")
    lines.append("读取 BlackHole 需要麦克风权限；未授权时系统会给静音。系统 tap 不需要这项权限。")
    lines.append("不会在系统采集失败时自动改用麦克风。")
    return lines
