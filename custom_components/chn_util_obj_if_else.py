import json
from langflow.custom import Component
from langflow.io import Output, DataInput, MultilineInput, DropdownInput
from langflow.schema import Data


class JsonFieldRouter(Component):
    display_name = "JSON Field Router (If-Else, Multi-Condition)"
    description = "รับ Data object แล้วตรวจสอบหลายเงื่อนไขพร้อมกัน (AND/OR) ส่ง object เต็มออกทาง TRUE หรือ FALSE"
    icon = "split"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Object",
            info="ลากสายจาก component ที่ส่งออกเป็น Data object มาเชื่อมที่นี่",
        ),
        MultilineInput(
            name="conditions_json",
            display_name="Conditions (JSON array)",
            info=(
                'ระบุเงื่อนไขหลายอันเป็น JSON array เช่น:\n'
                '[{"field": "is_safe", "operator": "equals", "value": "true"}, '
                '{"field": "score", "operator": "greater_than", "value": "80"}]\n'
                'operator ที่รองรับ: equals, not_equals, contains, greater_than, '
                'less_than, exists, not_exists\n'
                'field รองรับ nested แบบ user.profile.age'
            ),
            value='[\n  {"field": "", "operator": "equals", "value": ""}\n]',
        ),
        DropdownInput(
            name="logic_mode",
            display_name="Logic Mode",
            options=["AND", "OR"],
            value="AND",
            info="AND = ต้องผ่านทุกเงื่อนไข / OR = ผ่านแค่เงื่อนไขเดียวก็พอ",
        ),
    ]

    outputs = [
        Output(name="true_output", display_name="TRUE", method="true_response", group_outputs=True),
        Output(name="false_output", display_name="FALSE", method="false_response", group_outputs=True),
    ]

    def _get_value(self, data: dict, field_path: str):
        """ดึงค่าจาก dict ตาม path แบบ nested เช่น 'user.profile.age'"""
        value = data
        for key in field_path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _check_single_condition(self, value, operator: str, expected: str) -> bool:
        if operator == "exists":
            return value is not None
        if operator == "not_exists":
            return value is None
        if value is None:
            return False
        if operator == "equals":
            return str(value).strip().lower() == str(expected).strip().lower()
        if operator == "not_equals":
            return str(value).strip().lower() != str(expected).strip().lower()
        if operator == "contains":
            return str(expected) in str(value)
        if operator == "greater_than":
            return float(value) > float(expected)
        if operator == "less_than":
            return float(value) < float(expected)
        raise ValueError(f"ไม่รู้จัก operator: {operator}")

    def _parse_and_check(self) -> tuple[bool, Data]:
        # 1. รับข้อมูลจาก Data object ที่ต่อสายเข้ามา (ไม่ต้อง json.loads แล้ว)
        if self.input_data is None:
            raise ValueError("ไม่มี Input Object ต่อเข้ามา กรุณาลากสายจาก component ที่ส่งออกเป็น Data")

        data_obj = self.input_data if isinstance(self.input_data, Data) else Data(data=self.input_data)
        parsed = data_obj.data if isinstance(data_obj.data, dict) else {}

        # 2. Parse รายการเงื่อนไข
        try:
            conditions = json.loads(self.conditions_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Conditions ไม่ใช่ JSON array ที่ถูกต้อง: {e}") from e

        if not isinstance(conditions, list) or not conditions:
            raise ValueError("Conditions ต้องเป็น JSON array ที่มีอย่างน้อย 1 เงื่อนไข")

        # 3. ตรวจสอบทีละเงื่อนไข
        results = []
        for cond in conditions:
            field = cond.get("field", "")
            operator = cond.get("operator", "equals")
            expected = cond.get("value", "")
            value = self._get_value(parsed, field)
            check_result = self._check_single_condition(value, operator, expected)
            results.append(check_result)

        # 4. รวมผลตาม AND / OR
        final_result = all(results) if self.logic_mode == "AND" else any(results)

        return final_result, data_obj

    def true_response(self) -> Data:
        result, data_obj = self._parse_and_check()
        if not result:
            self.stop("true_output")
            return None
        self.status = data_obj
        return data_obj

    def false_response(self) -> Data:
        result, data_obj = self._parse_and_check()
        if result:
            self.stop("false_output")
            return None
        self.status = data_obj
        return data_obj