# SET + TFEX agent (โปรแกรมที่ 3)

ระบบเทรดหุ้นไทย (SET) และฟิวเจอร์ส (TFEX) — ตอนนี้อยู่ขั้น **backtest**

## ไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `config.py` | ช่วงเวลาทดสอบ, รายชื่อหุ้น, ค่าคอมมิชชั่น, เกณฑ์ผ่าน |
| `fetch_data.py` | ดึงราคาย้อนหลังจาก Yahoo ลง `data/` |
| `backtest.py` | ทดสอบ 3 ชุด: S50 Futures, Gold Futures (บาท), หุ้น momentum |
| `holdout.py` | เปิดผลช่วง 12 เดือนล่าสุด เฉพาะตัวที่ผ่านเกณฑ์ (เปิดครั้งเดียว) |
| `.github/workflows/backtest.yml` | ให้ GitHub รันทั้งหมดให้อัตโนมัติ |

## ข้อมูลที่ใช้แทนของจริง (proxy)
- S50 Futures → ดัชนี SET50 (ไม่รวมส่วนต่าง basis/ค่า rollover)
- Gold Futures → ราคาทอง COMEX × ค่าเงินบาท
- หุ้น → ราคาปรับปันผลจาก Yahoo, เฉพาะหุ้นที่ยังอยู่ในตลาด (มี survivorship bias)

เมื่อเปิดบัญชีโบรกที่มี Settrade Open API แล้ว จะเปลี่ยนไปใช้ข้อมูลจริงของ TFEX

## ย้ายไป VPS (Windows)
1. ติดตั้ง Python 3.12 และ Git
2. `git clone` repo นี้ไปที่ `C:\SetTfexAgent`
3. `pip install -r requirements.txt`
4. `python fetch_data.py` แล้ว `python backtest.py`

โค้ดใช้ path แบบสัมพัทธ์ทั้งหมด จึงรันบน VPS ได้โดยไม่ต้องแก้
