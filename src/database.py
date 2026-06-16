"""
Structured HR database access + LLM tool definitions.

Employee records live in a CSV loaded into a pandas DataFrame. The functions
here are exposed to the LLM as callable tools (OpenAI/OpenRouter "tools"
schema). The agent decides which tool to call; ``TOOL_EXECUTORS`` maps tool
names to the Python functions that run them.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


class EmployeeDB:
    """Read-only lookups over the employee CSV."""

    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or config.EMPLOYEE_DATA_PATH
        self.df = pd.read_csv(self.csv_path)
        logger.info("Loaded %d employee records", len(self.df))

    def list_employees(self) -> list[dict]:
        """Return id/name pairs for every employee (used to populate the UI)."""
        return self.df[["EmpID", "Name"]].to_dict("records")

    def _find(self, name_or_id: str) -> dict | None:
        """Match a row by exact EmpID or case-insensitive name substring."""
        key = str(name_or_id).strip()
        by_id = self.df[self.df["EmpID"].str.lower() == key.lower()]
        if not by_id.empty:
            return by_id.iloc[0].to_dict()
        by_name = self.df[self.df["Name"].str.contains(key, case=False, na=False)]
        if not by_name.empty:
            return by_name.iloc[0].to_dict()
        return None

    # --- Tool implementations -------------------------------------------------
    def get_employee(self, name_or_id: str) -> dict:
        """Full profile for one employee."""
        emp = self._find(name_or_id)
        if not emp:
            return {"error": f"No employee found matching '{name_or_id}'."}
        return {
            "EmpID": emp["EmpID"],
            "Name": emp["Name"],
            "Email": emp["Email"],
            "Phone": emp["Phone"],
            "Department": emp["Department"],
            "Role": emp["Role"],
            "Manager": emp["Manager"],
            "JoiningDate": emp["JoiningDate"],
        }

    def get_leave_balance(self, name_or_id: str) -> dict:
        """Leave balances for one employee."""
        emp = self._find(name_or_id)
        if not emp:
            return {"error": f"No employee found matching '{name_or_id}'."}
        casual, sick, earned = emp["CasualLeave"], emp["SickLeave"], emp["EarnedLeave"]
        return {
            "Name": emp["Name"],
            "CasualLeave": int(casual),
            "SickLeave": int(sick),
            "EarnedLeave": int(earned),
            "TotalLeave": int(casual + sick + earned),
        }

    def get_department(self, department: str) -> dict:
        """Roster and headcount for a department."""
        rows = self.df[self.df["Department"].str.contains(department, case=False, na=False)]
        if rows.empty:
            return {"error": f"No department matching '{department}'."}
        return {
            "Department": rows.iloc[0]["Department"],
            "TeamSize": len(rows),
            "Members": rows[["Name", "Role"]].to_dict("records"),
        }


# --- Tool schema exposed to the LLM ------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_employee",
            "description": "Look up an employee's profile (department, role, manager, contact, joining date) by name or employee ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_id": {
                        "type": "string",
                        "description": "Employee name or ID, e.g. 'John', 'Priya Sharma', or 'E001'.",
                    }
                },
                "required": ["name_or_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": "Get an employee's remaining leave balance (casual, sick, earned) by name or employee ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_id": {
                        "type": "string",
                        "description": "Employee name or ID.",
                    }
                },
                "required": ["name_or_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_department",
            "description": "List the members and headcount of an HR department, e.g. Engineering, HR, Finance, Marketing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Department name.",
                    }
                },
                "required": ["department"],
            },
        },
    },
]


def build_tool_executors(db: EmployeeDB) -> dict:
    """Map tool names to bound EmployeeDB methods."""
    return {
        "get_employee": db.get_employee,
        "get_leave_balance": db.get_leave_balance,
        "get_department": db.get_department,
    }


def run_tool(executors: dict, name: str, arguments: str) -> str:
    """Execute a tool call and return its result as a JSON string."""
    fn = executors.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments) if arguments else {}
        return json.dumps(fn(**args))
    except Exception as e:  # noqa: BLE001 - surface tool errors to the model
        logger.warning("Tool %s failed: %s", name, e)
        return json.dumps({"error": str(e)})
