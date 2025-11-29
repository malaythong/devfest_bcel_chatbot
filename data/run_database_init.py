
import asyncio
import csv
import vertexai
from datetime import datetime, time
from vertexai.language_models import TextEmbeddingModel

from toolbox_core import ToolboxClient
from agent.tools import TOOLBOX_URL

from models import Product

# ฟังก์ชันช่วยสร้าง Embedding (แปลงข้อความ -> ตัวเลข)
def get_embeddings(texts: list[str]) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    embeddings = model.get_embeddings(texts)
    return [e.values for e in embeddings]

async def load_dataset(products_ds_path: str) -> list[Product]:
    products: list[Product] = []
    
    with open(products_ds_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        rows = list(reader) 

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

        # 2. สร้างตารางใหม่ (Schema ต้องตรงกับ models.py)
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

        # 3. นำเข้าข้อมูล (Insert Data)
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

    products = await load_dataset(products_ds_path)
    await initialize_data(products)

    print("Database initialization complete.")

if __name__ == "__main__":
    asyncio.run(main())