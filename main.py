import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import face_recognition
import shutil
from PIL import Image
import numpy as np
import concurrent.futures
import threading
import pickle
import datetime
import torch  # GPU 사용을 위한 PyTorch 임포트
import multiprocessing

# GPU 사용 설정
if torch.cuda.is_available():
    device = "cuda"
    print("GPU를 사용합니다:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CPU를 사용합니다")

# CPU 코어 수 제한 설정
MAX_CPU_CORES = max(1, multiprocessing.cpu_count() - 1)  # 전체 코어 수에서 1개를 뺀 값 사용

class FaceRecognitionThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, known_faces_dir, target_dir, output_dir):
        super().__init__()
        self.known_faces_dir = known_faces_dir
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.known_faces = {}
        self.known_names = []
        self.batch_size = min(4, MAX_CPU_CORES)  # CPU 코어 수를 고려한 배치 크기 설정
        self.model_path = "face_encodings.pkl"
        
        # CPU 스레드 수 제한
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CPU_CORES)

    def save_encodings(self):
        """학습된 얼굴 특징점을 파일로 저장"""
        data = {
            'known_faces': self.known_faces,
            'known_names': self.known_names,
            'timestamp': datetime.datetime.now(),
            'source_dir': self.known_faces_dir
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n=== 얼굴 특징점 저장 완료 ===")
        print(f"저장 위치: {self.model_path}")
        print(f"저장된 인물 수: {len(self.known_names)}")

    def load_encodings(self):
        """저장된 얼굴 특징점 로드"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 저장된 데이터의 소스 디렉토리가 현재와 같은지 확인
                if data['source_dir'] == self.known_faces_dir:
                    self.known_faces = data['known_faces']
                    self.known_names = data['known_names']
                    print(f"\n=== 저장된 얼굴 특징점 로드 완료 ===")
                    print(f"로드된 인물 수: {len(self.known_names)}")
                    print(f"저장 시간: {data['timestamp']}")
                    return True
                else:
                    print("소스 디렉토리가 변경되어 재학습이 필요합니다.")
                    return False
            return False
        except Exception as e:
            print(f"저장된 특징점 로드 중 에러 발생: {str(e)}")
            return False

    def load_known_faces(self):
        # 저장된 특징점이 있는지 확인
        if self.load_encodings():
            return

        print("=== 학습 데이터 로딩 시작 ===")
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                encodings = []
                print(f"[{person_name}] 학습 이미지 처리 중...")
                for image_name in os.listdir(person_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir, image_name)
                        try:
                            image = face_recognition.load_image_file(image_path)
                            # 이미지 크기 조정으로 처리 속도 향상
                            height, width = image.shape[:2]
                            if width > 1024:
                                scale = 1024 / width
                                image = Image.fromarray(image)
                                image = image.resize((1024, int(height * scale)), Image.Resampling.LANCZOS)
                                image = np.array(image)
                            
                            # CNN 모델로 얼굴 위치 검출
                            face_locations = face_recognition.face_locations(image, model="cnn")
                            if face_locations:
                                # 각 얼굴에 대해 128개의 특징점 추출
                                face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=2)
                                if face_encodings:
                                    encodings.extend(face_encodings)
                                    print(f"  - {image_name}: 성공 (특징점 {len(face_encodings)}개 추출)")
                                else:
                                    print(f"  - {image_name}: 특징점 추출 실패")
                            else:
                                print(f"  - {image_name}: 얼굴 검출 실패")
                        except Exception as e:
                            print(f"  - {image_name}: 에러 발생 ({str(e)})")
                            self.error.emit(f"Error loading {image_path}: {str(e)}")
                
                if encodings:
                    self.known_faces[person_name] = encodings
                    self.known_names.append(person_name)
                    print(f"[{person_name}] 완료: {len(encodings)}개의 특징점 세트 저장")
                else:
                    print(f"[{person_name}] 경고: 추출된 특징점 없음")
        
        print("=== 학습 데이터 로딩 완료 ===")
        print(f"총 {len(self.known_names)}명의 인물 데이터 로드됨")

        # 추출된 특징점 저장
        self.save_encodings()

    def process_image(self, image_file):
        try:
            print(f"\n처리 중: {image_file}")
            image_path = os.path.join(self.target_dir, image_file)
            
            # 이미지 로드 및 크기 조정
            image = face_recognition.load_image_file(image_path)
            height, width = image.shape[:2]
            if width > 1024:
                scale = 1024 / width
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize((1024, int(height * scale)), Image.Resampling.LANCZOS)
                image = np.array(pil_image)

            # GPU 사용 가능 시 텐서로 변환
            if device == "cuda":
                image_tensor = torch.from_numpy(image).to(device)
                image = image_tensor.cpu().numpy()

            # CNN 모델로 얼굴 검출 (GPU 사용)
            print(f"얼굴 검출 중...")
            face_locations = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=1)
            if not face_locations:
                print("얼굴 검출 실패")
                shutil.copy2(image_path, os.path.join(self.output_dir, "미확인", image_file))
                return

            # 특징점 추출 (GPU 활용)
            face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=1)
            print(f"검출된 얼굴 수: {len(face_encodings)}")

            # 2명 이상인 경우
            if len(face_encodings) > 1:
                print("다수의 얼굴 발견")
                shutil.copy2(image_path, os.path.join(self.output_dir, "단체사진", image_file))
                return

            # 얼굴 매칭
            face_encoding = face_encodings[0]
            matches = {}
            
            # 각 인물과의 거리 계산
            for name, known_encodings in self.known_faces.items():
                # 여러 특징점과의 거리 계산
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                # 상위 3개의 가장 가까운 거리의 평균 사용
                top_3_distances = sorted(distances)[:3]
                avg_distance = sum(top_3_distances) / len(top_3_distances)
                matches[name] = avg_distance

            # 가장 가까운 인물 찾기
            best_match = min(matches.items(), key=lambda x: x[1])
            name, distance = best_match

            # 매칭 결과 출력
            print("\n=== 매칭 결과 ===")
            for person, dist in sorted(matches.items(), key=lambda x: x[1]):
                print(f"{person}: {dist:.3f}")
            print("================")

            # 거리가 임계값보다 작은 경우에만 매칭으로 판단
            if distance < 0.45:  # 더 엄격한 임계값 적용
                print(f"매칭 성공 -> {name} (거리: {distance:.3f})")
                shutil.copy2(image_path, os.path.join(self.output_dir, name, image_file))
            else:
                print(f"매칭 실패 (최소 거리: {distance:.3f})")
                shutil.copy2(image_path, os.path.join(self.output_dir, "미확인", image_file))

        except Exception as e:
            print(f"에러 발생: {str(e)}")
            self.error.emit(f"Error processing {image_file}: {str(e)}")

    def run(self):
        try:
            print("학습된 얼굴 로딩 시작")  # 디버깅용 로그
            # 학습된 얼굴 로드
            self.load_known_faces()
            print(f"로드된 인물: {self.known_names}")  # 디버깅용 로그

            # 출력 디렉토리 생성
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # 그룹 사진 디렉토리 생성
            group_dir = os.path.join(self.output_dir, "단체사진")
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

            # 미확인 인물 디렉토리 생성
            unknown_dir = os.path.join(self.output_dir, "미확인")
            if not os.path.exists(unknown_dir):
                os.makedirs(unknown_dir)

            # 각 인물별 디렉토리 생성
            for name in self.known_names:
                person_dir = os.path.join(self.output_dir, name)
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)

            # 대상 이미지 처리
            image_files = [f for f in os.listdir(self.target_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(image_files)
            print(f"처리할 이미지 수: {total_files}")  # 디버깅용 로그

            # 순차 처리로 변경
            for i, image_file in enumerate(image_files):
                try:
                    self.process_image(image_file)
                    self.progress.emit(int((i + 1) / total_files * 100))
                except Exception as e:
                    print(f"이미지 처리 중 에러 발생: {image_file} - {str(e)}")
                    continue

            print("처리 완료")  # 디버깅용 로그
            self.finished.emit()

        except Exception as e:
            print(f"전체 처리 중 에러 발생: {str(e)}")  # 디버깅용 로그
            self.error.emit(f"Error during processing: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("얼굴 인식 사진 분류 프로그램")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        # 메인 위젯과 레이아웃 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 학습 데이터 선택 버튼
        self.known_faces_btn = QPushButton("학습할 인물 사진 폴더 선택")
        self.known_faces_btn.clicked.connect(self.select_known_faces_dir)
        layout.addWidget(self.known_faces_btn)

        # 분류할 사진 선택 버튼
        self.target_btn = QPushButton("분류할 사진 폴더 선택")
        self.target_btn.clicked.connect(self.select_target_dir)
        layout.addWidget(self.target_btn)

        # 출력 폴더 선택 버튼
        self.output_btn = QPushButton("결과 저장 폴더 선택")
        self.output_btn.clicked.connect(self.select_output_dir)
        layout.addWidget(self.output_btn)

        # 상태 표시 레이블
        self.status_label = QLabel("폴더를 선택해주세요")
        layout.addWidget(self.status_label)

        # 진행바
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 시작 버튼
        self.start_btn = QPushButton("분류 시작")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)

        # 경로 저장 변수
        self.known_faces_dir = None
        self.target_dir = None
        self.output_dir = None

    def select_known_faces_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "학습할 인물 사진 폴더 선택")
        if dir_path:
            self.known_faces_dir = dir_path
            self.update_status()

    def select_target_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "분류할 사진 폴더 선택")
        if dir_path:
            self.target_dir = dir_path
            self.update_status()

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "결과 저장 폴더 선택")
        if dir_path:
            self.output_dir = dir_path
            self.update_status()

    def update_status(self):
        status = []
        if self.known_faces_dir:
            status.append(f"학습 폴더: {self.known_faces_dir}")
        if self.target_dir:
            status.append(f"대상 폴더: {self.target_dir}")
        if self.output_dir:
            status.append(f"출력 폴더: {self.output_dir}")

        self.status_label.setText("\n".join(status))
        
        # 모든 폴더가 선택되었는지 확인
        self.start_btn.setEnabled(all([self.known_faces_dir, self.target_dir, self.output_dir]))

    def start_processing(self):
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        
        # 처리 스레드 시작
        self.thread = FaceRecognitionThread(self.known_faces_dir, self.target_dir, self.output_dir)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self):
        self.status_label.setText("처리가 완료되었습니다!")
        self.start_btn.setEnabled(True)

    def show_error(self, error_message):
        self.status_label.setText(f"에러: {error_message}")
        self.start_btn.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
