from langflow.custom import Component
from langflow.io import Output, DataInput, IntInput
from langflow.schema import Data


class ObjectMerger(Component):
    display_name = "Object Merger (Dynamic Count)"
    description = (
        "ระบุจำนวน Object ที่ต้องการรวม ระบบจะสร้างช่องรับ Object ให้ตามจำนวนนั้นอัตโนมัติ "
        "ลำดับการ merge เรียงจาก Object 1 → 2 → 3 ... ตัวหลังทับตัวก่อนหน้าถ้า field ซ้ำ"
    )
    icon = "merge"
    MAX_OBJECTS = 50  # กันไม่ให้ใส่เลขเยอะเกินจนช้า/รก

    inputs = [
        IntInput(
            name="num_objects",
            display_name="Number of Objects",
            info="ระบุจำนวน Object ที่ต้องการรวม แล้วคลิกออกนอกช่อง ระบบจะสร้างช่องรับให้ตามจำนวนนี้",
            value=2,
            real_time_refresh=True,
        ),
    ]

    outputs = [
        Output(
            name="merged_output",
            display_name="Merged Object",
            method="merge_objects",
        ),
    ]

    def update_build_config(self, build_config: dict, field_value, field_name: str | None = None) -> dict:
        if field_name != "num_objects":
            return build_config

        # ลบช่อง Object เดิมทั้งหมดก่อน (รองรับกรณีลดจำนวนลง)
        for key in [k for k in build_config if k.startswith("input_object_")]:
            del build_config[key]

        try:
            count = int(field_value) if field_value else 0
        except (TypeError, ValueError):
            count = 0
        count = max(0, min(count, self.MAX_OBJECTS))

        # สร้างช่องใหม่ตามจำนวนที่ระบุ
        for i in range(1, count + 1):
            key = f"input_object_{i}"
            build_config[key] = DataInput(
                name=key,
                display_name=f"Object {i}",
                info=f"ลำดับที่ {i} — field ซ้ำจะถูกทับโดย Object ที่ลำดับสูงกว่า",
            ).to_dict()

        return build_config

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Merge แบบ recursive: dict ซ้อนกันจะ merge ลึกลงไป ค่าอื่นตัวหลังทับตัวก่อนเสมอ"""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def merge_objects(self) -> Data:
        count = int(self.num_objects) if self.num_objects else 0
        merged: dict = {}
        has_any = False

        for i in range(1, count + 1):
            item = getattr(self, f"input_object_{i}", None)
            if item is None:
                continue
            has_any = True
            source = item if isinstance(item, Data) else Data(data=item)
            obj = source.data if isinstance(source.data, dict) else {}
            merged = self._deep_merge(merged, obj)

        if not has_any:
            raise ValueError("ไม่มี Input Object ต่อเข้ามาเลย กรุณาต่อสายอย่างน้อย 1 ช่อง")

        output = Data(data=merged)
        self.status = output
        return output