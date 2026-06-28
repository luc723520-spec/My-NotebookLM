# Mini NotebookLM & Quiz Studio 

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLM-black?style=for-the-badge)

Một hệ thống EdTech Full-Stack ứng dụng kiến trúc **Retrieval-Augmented Generation (RAG)** nâng cao, được thiết kế theo phong cách không gian làm việc hợp nhất (Unified Workspace Layout) lấy cảm hứng từ Google NotebookLM. 

Hệ thống cho phép người dùng nạp tài liệu học thuật một lần duy nhất, sau đó khai thác toàn diện thông qua 3 phân hệ tương tác song song: **Hỏi đáp chuyên sâu**, **Sinh đề thi trắc nghiệm tự động**, và **Trực quan hóa bản đồ tư duy**.



##  Privacy-First & 100% Offline
Dự án được xây dựng dựa trên nguyên tắc bảo mật dữ liệu tuyệt đối. Toàn bộ quy trình từ tải tệp cấu trúc, trích xuất dữ liệu vector, phân loại ý định người dùng cho đến chuỗi suy luận logic cốt lõi đều được xử lý hoàn toàn **cục bộ (cục bộ Offline)** trên thiết bị cá nhân thông qua **Ollama** và **ChromaDB**. Không có bất kỳ byte dữ liệu nào bị gửi ra internet, đảm bảo an toàn tuyệt đối cho các tài liệu nghiên cứu nội bộ và học thuật.



##  Các Tính năng Công nghệ Nổi bật

### 1. Phân hệ Notebook Studio (Module A)
* **Xử lý hội thoại thông minh (RAG):** Trích xuất thông tin thông minh từ tệp PDF và trả lời nghiêm ngặt dựa trên ngữ cảnh để triệt tiêu hiện tượng ảo giác (Zero-Hallucination).
* **Định tuyến nhiệt độ tự động (Adaptive Temperature Routing):** Hệ thống tích hợp một bộ phân loại ý định (Zero-Shot Intent Classifier) chạy ngầm để nhận diện bản chất câu hỏi và tự điều chỉnh hệ số sáng tạo `temperature` theo thời gian thực:
  * *Nhóm 1 (Hỏi đáp dữ liệu gốc):* Đặt `temperature = 0.0` để tối ưu tính chính xác tuyệt đối.
  * *Nhóm 2 (Phân tích mở rộng thực tế):* Đặt `temperature = 0.3` để dung hòa giữa lý thuyết và ứng dụng.
  * *Nhóm 3 (Trò chuyện tự do ngoài lề):* Đặt `temperature = 0.5` để tối ưu hóa sự linh hoạt ngôn từ.
* **Suy luận chuỗi (Chain-of-Thought):** Ép mô hình thực hiện các bước lập luận logic đa chiều bên trong khối thẻ ẩn `<thinking>` trước khi đưa ra câu trả lời cuối cùng để nâng cao chất lượng phản hồi đối với các tài liệu phức tạp.
* **Hỗ trợ đa phương tiện:** Tích hợp bộ chuyển đổi văn bản thành giọng đọc Text-to-Speech (TTS) tự động mã hóa Base64 Audio phát trực tiếp trên UI.

### 2. Phân hệ Auto-Quiz Generator (Module B)
* **Luồng xử lý 1-Chạm:** Tự động hóa toàn bộ quy trình từ việc quét cơ sở dữ liệu vector đến việc ép cấu hình Prompt tinh chỉnh (Few-Shot Prompting) nhằm ép mô hình xuất ra cấu trúc dữ liệu JSON khắt khe, đẩy thẳng vào State của React làm bài tập ngay lập tức.
* **Định hướng mục tiêu (Directional Stimulus Prompting):** Cho phép người dùng nhập từ khóa trọng tâm hoặc yêu cầu độ khó (Ví dụ: *"Khó, tập trung vào tính toán"*), hệ thống sẽ hướng dẫn mô hình biên soạn bộ đề bám sát mục tiêu sư phạm.

### 3. Phân hệ Mindmap Visualizer (Module C)
* **Tự động hóa sơ đồ cây (Auto-Trigger):** Ngay khi người dùng chuyển sang phân hệ Sơ đồ tư duy, hệ thống sẽ tự động phân tích cấu trúc tổng quan tài liệu để bóc tách từ 3-5 luận điểm cốt lõi nhất.
* **Render thời gian thực:** Ép cấu trúc mô hình sinh mã ngôn ngữ đồ thị cấu trúc **Mermaid.js**, kết hợp thư viện frontend để biên dịch trực quan hóa thành một bản đồ tư duy dạng cây ngay trên màn hình.

---

## 📂 Cấu trúc thư mục (Project Structure)

```text
Day5-Notebook_and_quiz/
│
├── 📁 quiz-app/             
│   ├── 📁 src/               
│   │   ├── 📁 components/   
│   │   ├── 📁 lib/           
│   │   ├── App.jsx           
│   │   └── main.jsx
│   ├── package.json          
│   └── vite.config.js        
│
├── 📁 src/                  
│   ├── apply-knowledge.py    # Ứng dụng Streamlit hỗ trợ kiểm thử tính năng độc lập
│   └── backend.py            # API Server FastAPI (Xử lý Core RAG & AI Routing)
│
├── .gitignore              
└── requirements.txt          # Danh sách các thư viện
---
### 2. Phần Hướng dẫn cài đặt (Local Setup)
Sử dụng khối ````bash ```` để tự động tô màu các câu lệnh Terminal, giúp người đọc dễ dàng copy lệnh để chạy:

```markdown
 Hướng dẫn Cài đặt & Khởi chạy Cục bộ (Local Setup)
Để chạy dự án này trên máy tính cá nhân của bạn sau khi kéo (pull) mã nguồn về, hãy thực hiện chính xác theo quy trình 3 giai đoạn dưới đây:

Giai đoạn 1: Chuẩn bị Môi trường Tiền quyết
Đảm bảo máy tính đã cài đặt Node.js (Phiên bản >= 18.x) và Python (Phiên bản 3.11 hoặc 3.12).

Tải và cài đặt nền tảng Ollama phù hợp với hệ điều hành của bạn.

Mở Terminal hệ thống lên và kéo mô hình Llama 3.2 về máy bằng cách chạy lệnh:

Bash
ollama run llama3.2
Kiểm tra đảm bảo mô hình đã tải xong và chạy ổn định trong Terminal trước khi sang bước tiếp theo.

 Giai đoạn 2: Cấu hình và Chạy Backend (FastAPI)
Mở một cửa sổ Terminal mới và di chuyển vào thư mục gốc của dự án (Day5-Notebook_and_quiz):

Bash
# 1. Tạo môi trường ảo Python (.venv)
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Đối với MacOS/Linux/WSL:
source .venv/bin/activate
# Đối với Windows (CMD / PowerShell):
# .venv\Scripts\activate

# 3. Nâng cấp bộ cài đặt pip và cài các thư viện AI cần thiết
pip install --upgrade pip
pip install -r requirements.txt

# 4. Khởi chạy máy chủ API FastAPI
uvicorn src.backend:app --reload --host 127.0.0.1 --port 8000
Khi thấy hệ thống thông báo Application startup complete. Uvicorn running on http://127.0.0.1:8000 nghĩa là cổng dịch vụ Backend đã thông suốt.

Giai đoạn 3: Cấu hình và Chạy Frontend (React)
Mở một cửa sổ Terminal khác song song và di chuyển vào phân hệ giao diện:

Bash
# 1. Di chuyển vào thư mục frontend
cd quiz-app

# 2. Cài đặt toàn bộ các gói thư viện Node.js (Bao gồm cả Tailwind và Mermaid)
npm install

# 3. Khởi chạy giao diện nhà phát triển trên môi trường cục bộ
npm run dev
