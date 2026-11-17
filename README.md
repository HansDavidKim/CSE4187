### 규칙
- 각자의 기능은 feat/기능명 브랜치에서 개발
    - 고장은 fix/기능명
- conflict 없을 시 메인에 병합알아서. conflict 나면 카톡방에 보고
- 웬만하면 자신의 디렉토리에서 작업. 
- docs, common은 필요하다 싶으면 추가 가능.

### 시뮬레이터

시뮬레이터 코드·설정·테스트는 [simulator/](simulator/README.md)에서 관리한다.
기존 `simulator_etc/`의 실행 코드는 이 디렉터리로 통합했다.

```bash
cd simulator
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```
