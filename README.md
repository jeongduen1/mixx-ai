---
marp: true
theme: default
class: lead
---

# 얼굴 인식 기반 사진 분류  
## MIXX AI
(PyQt6 + face_recognition)
30819 정찬영

---

## 사용한 AI 모델

- 라이브러리: `face_recognition`  
- 내부 모델: **Dlib** 기반 **ResNet-34 CNN**  
- 기능:
  - 얼굴 검출 (`face_locations`)
  - 얼굴 특징 벡터 추출 (`face_encodings`)

> 별도 모델 설계 없이도 고성능 얼굴 인식 기능 사용 가능

---

## 주요 모듈 요약

| 모듈 | 기능 |
|------|------|
| `face_recognition` | 얼굴 검출 및 특징 추출 |
| `PyQt6` | GUI 제작 |
| `Pillow`, `numpy` | 이미지 전처리 |
| `pickle` | 얼굴 벡터 저장/불러오기 |
| `concurrent.futures` | (예정) 병렬 처리 |
| `torch` | GPU 사용 여부 판단 용도 |

---
<!-- style: h1{padding-top: 20%} -->
# AI 학습 방식

---
## 특징 추출

```python
image = face_recognition.load_image_file(image_path)
face_locations = face_recognition.face_locations(image, model="cnn")
face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=2)
```
> 각 이미지에서 128차원의 벡터를 추출하여 저장함
> 학습한 특징은 .pkl로 저장됨

---

# 실제 분류 과정

---


## 1. 대상 이미지에서 얼굴 검출 및 특징 추출
```python
image = face_recognition.load_image_file(target_path)
locations = face_recognition.face_locations(image, model="cnn")
encodings = face_recognition.face_encodings(image, locations)
```

---

## 2. 모든 학습된 사람과 거리 계산
```python
for name, known_encs in known_faces.items():
    distances = face_recognition.face_distance(known_encs, target_encoding)
    avg = sum(sorted(distances)[:3]) / 3
```

---

## 3. 최솟값이 임계값보다 작으면 매칭 성공
```python
if best_avg_distance < 0.45:
    분류 성공 → 해당 폴더로 복사
else:
    미확인 또는 단체사진으로 분류
```

---


## 정리

face_recognition은 CNN 기반 얼굴 검출 및 벡터 추출 기능 제공

벡터 비교만으로 인물 매칭이 가능

.pkl 캐싱으로 재학습 없이 빠른 분류 지원

정확도: 93.33% (60개 중 56개 정답)

---

# 감사합니다
