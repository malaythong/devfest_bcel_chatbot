# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

WORKDIR /app

# 1. ติดตั้ง Curl (เพื่อโหลด Toolbox)
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# 2. ดาวน์โหลด Toolbox (Server)
# ระบุเวอร์ชัน v0.5.0 ให้ชัดเจน เพื่อแก้ปัญหา NoSuchKey ที่คุณเคยเจอ
RUN curl -o toolbox https://storage.googleapis.com/genai-toolbox/v0.5.0/linux/amd64/toolbox && \
    chmod +x toolbox

# 3. ติดตั้ง Python Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy โค้ดทั้งหมด (รวมถึง run_app.py ด้วย)
COPY . .

# 5. สร้าง Script เพื่อรัน 2 อย่างพร้อมกัน (Toolbox + App)
# แก้ไขบรรทัดสุดท้ายให้เรียก 'python run_app.py' แทน uvicorn command
RUN echo '#!/bin/bash \n\
echo "Starting Toolbox..." \n\
./toolbox --tools-file tools.yaml --port 5000 & \n\
sleep 5 \n\
echo "Starting Web App..." \n\
export TOOLBOX_URL="http://127.0.0.1:5000" \n\
python run_app.py' > start.sh && chmod +x start.sh

# 6. ตั้งค่า Port (Cloud Run จะส่ง Port 8080 มาให้)
ENV PORT=8080
EXPOSE 8080

# 7. สั่งเริ่มทำงาน
CMD ["./start.sh"]
