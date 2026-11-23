@echo off
chcp 65001 >nul
setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PY=%SCRIPT_DIR%..\.venv\Scripts\python.exe"

if "%~1"=="" (
  echo ============================================================
  echo   SmartScout AI 電腦端高精度批次產片工具
  echo ============================================================
  echo 使用方式：直接把「單支影片 (.mp4/.mov)」或「整個牛棚資料夾」
  echo           拖曳到此 .bat 檔案上放開，就會自動產出 3D 紅軌跡
  echo           ＋ 8 項數據卡片的 SmartScout 短片與 CSV 報表！
  echo.
  pause
  exit /b 0
)

if exist "%VENV_PY%" (
  "%VENV_PY%" "%SCRIPT_DIR%smartscout_studio.py" %*
) else (
  python "%SCRIPT_DIR%smartscout_studio.py" %*
)

echo.
if exist "%SCRIPT_DIR%output" start "" "%SCRIPT_DIR%output"
pause
