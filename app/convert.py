import shutil
import subprocess
from pathlib import Path
import re


INPUT_DIR = Path("/home/shared/files")
OUTPUT_DIR = Path("/home/bidcoin/output_pdf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LIBREOFFICE_BIN = shutil.which("soffice") or shutil.which("libreoffice") or "soffice"


def prepare_pdf(input_path: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = input_path.suffix.lower()

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_path}")

    # PDF면 그대로 복사
    if ext == ".pdf":
        output_pdf_path = output_dir / f"{input_path.stem}.pdf"
        shutil.copy2(input_path, output_pdf_path)
        return output_pdf_path

    # HWP면 변환
    elif ext == ".hwp":
        before_pdfs = set(output_dir.glob("*.pdf"))

        cmd = [
            LIBREOFFICE_BIN,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(input_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            raise RuntimeError(
                f"HWP -> PDF 변환 실패\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        # 1순위: stdout에서 실제 생성 파일명 파싱
        match = re.search(r"->\s(.+?\.pdf)\susing filter", stdout)
        if match:
            pdf_path = Path(match.group(1))
            if not pdf_path.is_absolute():
                pdf_path = output_dir / pdf_path.name
            if pdf_path.exists():
                return pdf_path

        # 2순위: 예상 파일명 확인
        expected_pdf = output_dir / f"{input_path.stem}.pdf"
        if expected_pdf.exists():
            return expected_pdf

        # 3순위: 변환 직후 새로 생긴 pdf 찾기
        after_pdfs = set(output_dir.glob("*.pdf"))
        new_pdfs = list(after_pdfs - before_pdfs)
        if len(new_pdfs) == 1:
            return new_pdfs[0]

        raise FileNotFoundError(
            f"변환된 PDF를 찾을 수 없습니다.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


if __name__ == "__main__":
    for file_path in INPUT_DIR.iterdir():
        try:
            result_path = prepare_pdf(file_path, OUTPUT_DIR)
            print(f"완료: {file_path} -> {result_path}")
        except Exception as e:
            print(f"실패: {file_path}\n{e}")