# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import csv
import vertexai
from datetime import datetime, time
from vertexai.language_models import TextEmbeddingModel

# ตรวจสอบ import นี้ให้ถูกต้องตามโปรเจกต์เดิม
from toolbox_core import ToolboxClient
from agent.tools import TOOLBOX_URL

# Import Model Product ที่เราสร้างกันในขั้นตอนก่อนหน้า
from models import Product

# ฟังก์ชันช่วยสร้าง Embedding (แปลงข้อความ -> ตัวเลข)
def get_embeddings(texts: list[str]) -> list[list[float]]:
    # ใช้ Model text-embedding-004 (รุ่นใหม่ รองรับหลายภาษา)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    # Vertex AI รับ input ได้เป็น batch (แนะนำทีละไม่เกิน 5-10 ถ้าข้อมูลเยอะ)
    embeddings = model.get_embeddings(texts)
    return [e.values for e in embeddings]

async def load_dataset(products_ds_path: str) -> list[Product]:
    products: list[Product] = []
    
    # อ่านไฟล์ CSV (ระบุ encoding='utf-8-sig' เพื่อรองรับภาษาลาว)
    with open(products_ds_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        rows = list(reader)  # อ่านทั้งหมดมาเก็บใน list ก่อน

        # เตรียมข้อความสำหรับทำ Embedding
        # เราจะเอา "ชื่อสินค้า + รายละเอียด + กลุ่มลูกค้า" มารวมกันเพื่อให้ AI เข้าใจบริบทครบถ้วน
        texts_to_embed = [
            f"{row['product_name']} {row['description']} ({row['audience']})" 
            for row in rows
        ]

        # สร้าง Embeddings (ถ้าข้อมูลเยอะ ควรแบ่งทำทีละ batch)
        print(f"Generating embeddings for {len(rows)} products...")
        vectors = get_embeddings(texts_to_embed)

        for i, row in enumerate(rows):
            # แก้ไขชื่อ Key ให้ตรงกับใน models.py (CSV เป็น P ตัวใหญ่)
            # ถ้า models.py คุณใช้ 'products_types' แต่ CSV เป็น 'Products_types'
            row['products_types'] = row.pop('Products_types', '')

            # ใส่ค่า Embedding ที่สร้างเสร็จแล้วกลับเข้าไป
            row['embedding'] = vectors[i]
            
            # แปลงเป็น Object Product
            products.append(Product.model_validate(row))

    return products

def __escape_sql(value):
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return f"""'{value.replace("'", "''")}'"""
    # List (Vector) จะถูกแปลงเป็น string เช่น '[0.1, 0.2, ...]' ซึ่ง SQL รับได้
    if isinstance(value, list):
        return f"""'{value}'"""
    return value

async def initialize_data(products: list[Product]) -> None:
    async with ToolboxClient(TOOLBOX_URL) as toolbox:
        execute_sql = await toolbox.load_tool("execute_sql")

        print("Initializing database...")

        # 1. เปิดใช้งาน Vector Extension
        await execute_sql("CREATE EXTENSION IF NOT EXISTS vector")

        # 2. ลบตารางเก่าทิ้ง (Clean up)
        await execute_sql("DROP TABLE IF EXISTS airports CASCADE")
        await execute_sql("DROP TABLE IF EXISTS amenities CASCADE")
        await execute_sql("DROP TABLE IF EXISTS flights CASCADE")
        await execute_sql("DROP TABLE IF EXISTS tickets CASCADE")
        await execute_sql("DROP TABLE IF EXISTS policies CASCADE")
        
        # ลบตาราง products เดิมด้วย (ถ้ามี)
        await execute_sql("DROP TABLE IF EXISTS products CASCADE")

        # 3. สร้างตารางใหม่ (Schema ต้องตรงกับ models.py)
        # embedding vector(768) คือขนาดมาตรฐานของ text-embedding-004
        await execute_sql(
            """
            CREATE TABLE products(
                product_id TEXT PRIMARY KEY,
                product_name TEXT,
                description TEXT,
                type TEXT,
                status TEXT,
                audience TEXT,
                products_types TEXT,
                installation TEXT,
                embedding vector(768) 
            )
        """
        )

        # 4. นำเข้าข้อมูล (Insert Data)
        if not products:
            print("No products found to insert.")
            return

        values = [
            f"""(
            {__escape_sql(p.product_id)},
            {__escape_sql(p.product_name)},
            {__escape_sql(p.description)},
            {__escape_sql(p.type)},
            {__escape_sql(p.status)},
            {__escape_sql(p.audience)},
            {__escape_sql(p.products_types)},
            {__escape_sql(p.installation)},
            {__escape_sql(p.embedding)}
        )"""
            for p in products
        ]
        
        # Insert ทีเดียวทั้งหมด
        await execute_sql(f"""INSERT INTO products VALUES {", ".join(values)}""")
        print(f"Successfully inserted {len(products)} products into 'products' table.")

async def main() -> None:
    # อย่าลืมเปลี่ยนชื่อไฟล์ CSV ให้ตรงกับที่คุณเซฟไว้ในโฟลเดอร์ data/
    products_ds_path = "data/bcel.csv" 
    # หรือ products_ds_path = "data/products.csv" ถ้าคุณเปลี่ยนชื่อไฟล์แล้ว

    # Initialize Vertex AI (ถ้า environment ไม่ได้ auto-set)
    # vertexai.init(project="YOUR-PROJECT-ID", location="us-central1")

    products = await load_dataset(products_ds_path)
    await initialize_data(products)

    print("Database initialization complete.")

if __name__ == "__main__":
    asyncio.run(main())