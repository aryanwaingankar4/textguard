# Export paper/TextGuard_paper.docx to paper/TextGuard_paper.pdf with Microsoft Word (COM).
# Word lays the document out exactly as a reader opening the .docx will see it, so the PDF is a
# faithful preview of the Word layout. Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File paper\export_pdf.ps1
# A hidden Word instance cannot show a dialog; if Word ever hangs here, open the .docx once in
# Word by hand (to clear a recovery or safe-mode prompt) and run the script again.
param([string]$Dir = $PSScriptRoot)
$src = Join-Path $Dir "TextGuard_paper.docx"
$dst = Join-Path $Dir "TextGuard_paper.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($src, $false, $true, $false)   # no conversion prompt, read-only, not in MRU
    $doc.ExportAsFixedFormat($dst, 17)                         # 17 = wdExportFormatPDF
    $pages = $doc.ComputeStatistics(2)                         # 2 = wdStatisticPages
    $doc.Close($false)
    Write-Output "$dst  ($pages pages)"
} finally {
    $word.Quit()
}
