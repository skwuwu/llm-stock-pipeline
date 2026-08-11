# 일별 시세 갱신 — Windows 작업 스케줄러용 래퍼.
#
# 등록 (관리자 PowerShell 불필요, 사용자 계정으로 충분):
#   $repo = "C:\Users\gimgy\OneDrive\바탕 화면\LLM_stock_pipeline"
#   $act  = New-ScheduledTaskAction -Execute "powershell.exe" `
#             -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\daily_prices.ps1`""
#   $trg  = New-ScheduledTaskTrigger -Daily -At 16:10
#   Register-ScheduledTask -TaskName "LLM-stock daily prices" -Action $act -Trigger $trg
#
# 16:10 인 이유: 정규장 마감 15:30 + 종가 확정·데이터 반영 여유.
# 주말·공휴일에 실행돼도 스크립트가 마지막 거래일을 스스로 찾아 이미 받았으면
# 건너뛴다. 그래서 트리거에 요일 조건을 걸지 않는다 —
# 조건을 두 군데(스케줄러와 스크립트)에 두면 임시휴장 때 어긋난다.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 한글 로그가 깨지는 것을 막는다. **양쪽 다** 필요하다:
#   PYTHONIOENCODING — 파이썬이 UTF-8 로 내보내게
#   Console::OutputEncoding — PowerShell 이 그 바이트를 cp949 로 오독하지 않게
# 하나만 설정하면 파일에는 깨진 글자가 남는다(실측).
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8

$log = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$out = Join-Path $log ("run_" + (Get-Date -Format "yyyy-MM-dd") + ".txt")

# --derive 까지 하는 이유: 시세만 갱신하고 파생을 안 돌리면
# metrics_{date}.parquet 이 없어 screen 이 그날 값을 못 본다.
# --scan 은 바스켓을 **갱신하지 않고** 진입·이탈 후보만 보고한다.
# 바스켓 확정(리밸런스)은 사람이 판단해 주/월 단위로 따로 돌린다:
#     python -m pipeline.cli screen --as-of <date> --rebalance
#     이어서 enrich / tag / verify / golden
& python "scripts\daily_prices.py" --workers 16 --derive --scan *>&1 | Tee-Object -FilePath $out -Append
$rc = $LASTEXITCODE

if ($rc -ne 0) {
    # 스케줄러가 실패를 기록하게 종료 코드를 그대로 넘긴다.
    Write-Error "daily_prices 실패 (exit $rc) — $out 확인"
    exit $rc
}
exit 0
