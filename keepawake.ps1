# Holds the system awake (display may still turn off) while a target process
# lives. Usage: powershell -File keepawake.ps1 -TargetPid <pid>
param([Parameter(Mandatory = $true)][int]$TargetPid)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
        # re-assert; harmless if already set
        [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
    }
}
finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
