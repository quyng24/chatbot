# Learn AI

Dự án chatbot AI được xây dựng theo từng phase để thực hành Next.js, streaming, tool calling, RAG và tối ưu hiệu năng.

## Bắt đầu

### Yêu cầu

- Node.js tương thích với Next.js 16
- npm
- Gemini API key

### Cài đặt và chạy local

```bash
npm install
```

Sao chép file mẫu `.env.example` thành `.env.local`, sau đó điền Gemini API key của bạn:

```env
GOOGLE_GENERATIVE_AI_API_KEY=your_gemini_api_key
```

Không commit hoặc push `.env.local`; file này chứa API key cá nhân.

Khởi chạy ứng dụng:

```bash
npm run dev
```

Mở [http://localhost:3000](http://localhost:3000).

### Các lệnh thường dùng

```bash
npm run dev    # Chạy môi trường phát triển
npm run lint   # Kiểm tra ESLint
npx tsc --noEmit # Kiểm tra TypeScript
npm run build  # Tạo bản build production
npm run start  # Chạy bản production sau khi build
```

## Tiến độ và lộ trình

### Phase 1: Chatbot cơ bản — Hoàn thành

Mục tiêu: xây dựng ứng dụng chat có giao diện responsive và nhận câu trả lời AI theo dạng streaming.

- Khởi tạo ứng dụng bằng Next.js App Router, Tailwind CSS và các UI component theo Shadcn.
- Tạo API route tại `src/app/api/chat/route.ts`, kết nối Gemini bằng AI SDK.
- Dùng `useChat` để gửi tin nhắn và nhận nội dung streaming.
- Hiển thị Markdown trong câu trả lời; xử lý trạng thái đang trả lời và lỗi quota, bao gồm thời gian có thể thử lại.
- Hoàn thiện giao diện chat thích ứng cho desktop và mobile.
- Deploy trên Vercel; cấu hình `GOOGLE_GENERATIVE_AI_API_KEY` trong Environment Variables của project.

### Phase 2: Tool Calling và AI UI — Tiếp theo

Mục tiêu: để AI gọi công cụ và hiển thị kết quả dưới dạng giao diện tương tác.

- Tạo khu vực Canvas ở bên phải khung chat, tối ưu bố cục cho màn hình nhỏ.
- Tìm hiểu và cấu hình Tool Calling hoặc `streamUI` phù hợp với phiên bản AI SDK đang dùng.
- Xây dựng công cụ mẫu để AI có thể yêu cầu hiển thị biểu đồ hoặc giao diện thời tiết.
- Định nghĩa schema JSON có kiểm tra kiểu và validation cho dữ liệu từ AI.
- Render React component theo loại dữ liệu đã được xác thực; không render trực tiếp mã tùy ý do AI sinh ra.
- Tách các component nặng và lazy load khi cần.

### Phase 3: RAG — Dữ liệu và truy xuất ngữ cảnh

Mục tiêu: cho phép chatbot trả lời dựa trên tài liệu do người dùng cung cấp.

- Thêm chức năng upload PDF và trích xuất nội dung bằng thư viện phù hợp, ví dụ `pdf-parse` nếu tương thích môi trường Node.js của dự án.
- Tạo project Supabase và cấu hình thông tin kết nối qua biến môi trường.
- Chia nội dung tài liệu thành các chunk phù hợp; tạo embeddings và lưu nội dung cùng vector vào database.
- Khi nhận câu hỏi, tìm các chunk liên quan rồi đưa chúng vào context trước khi gọi model.
- Xử lý quyền truy cập, giới hạn kích thước file, lỗi parse và vòng đời tài liệu.

### Phase 4: Tối ưu hóa — State, lưu trữ và hiệu năng

Mục tiêu: cải thiện khả năng sử dụng khi ứng dụng có nhiều trạng thái và lịch sử chat.

- Dùng Zustand để quản lý trạng thái layout, chẳng hạn đóng/mở Canvas.
- Dùng PostgreSQL/Supabase để lưu hội thoại và lịch sử chat.
- Tối ưu React re-render và tách component theo trách nhiệm.
- Lazy load các component nặng và đo hiệu quả trước/sau khi tối ưu.
- Rà soát trạng thái tải, lỗi, empty state và trải nghiệm trên mobile.

## Cấu trúc hiện tại

```text
src/
├── app/
│   ├── api/chat/route.ts       # API streaming đến Gemini
│   ├── globals.css             # Global styles và theme
│   ├── layout.tsx              # Root layout
│   └── page.tsx                # Trang chính
├── components/
│   ├── loading/                # Loading skeleton và typing indicator
│   ├── ui/                     # UI primitives
│   └── ChatUI.tsx              # Giao diện chat và useChat
├── constants/
│   └── message.ts              # System prompt
└── lib/
    └── chat-error.ts           # Chuẩn hóa lỗi từ provider AI
```

## Công nghệ

- Next.js 16, React 19, TypeScript
- Tailwind CSS
- AI SDK và Google Gemini
- React Markdown
- UI primitives theo Shadcn/Base UI
- Vercel (mục tiêu triển khai)
