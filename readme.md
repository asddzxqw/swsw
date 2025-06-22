# YouTube 댓글 AI 분석기

YouTube 댓글을 AI로 분석하여 여론을 실시간 모니터링하는 시스템

## 주요 기능

- **AI 감정 분석**: 댓글을 5가지 감정으로 자동 분류
- **실시간 모니터링**: 부정적 여론 급증 시 위험 경보
- **시각화**: 차트와 그래프로 분석 결과 표시
- **키워드 분석**: 주요 키워드 자동 추출

## 설치 및 실행

```bash
git clone https://github.com/your-username/youtube-comment-analyzer.git
cd youtube-comment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## 필요한 것

- Python 3.8+
- YouTube Data API 키

## 사용법

1. YouTube API 키 입력
2. YouTube 동영상 URL 입력
3. 분석할 댓글 수 선택 (50-500개)
4. 분석 시작

## 감정 분류

- **찬성/지지**: 긍정적 반응
- **반대/비판**: 부정적 반응  
- **질문/요청/정보성**: 질문이나 정보 요청
- **비꼬기/유머/감탄**: 유머나 비꼼
- **중립/기타**: 중립적 의견

## 위험도 기준

- 🚨 **50% 이상**: 위험 (즉시 대응 필요)
- ⚠️ **30-50%**: 경고 (주의 필요) 
- ✅ **30% 미만**: 안정

## 기술 스택

- Python, Streamlit, Transformers, PyTorch
- YouTube Data API v3
- AI 모델: RoBERTa, BERT, DistilBERT

## 팀

- 김민성 (팀장)
- 김창준  
- 이정환