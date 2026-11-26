"""Summarize reported usage without treating unavailable counts or costs as zero."""

from collections import defaultdict
from decimal import Decimal


def summarize_usage(records):
    records = list(records)
    summary = {"calls": len(records), "reported_calls": {}}
    for field in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
                  "reasoning_tokens", "cost_usd"):
        values = []
        for row in records:
            value = row.get(field)
            if field == "total_tokens" and value is None:
                if row.get("input_tokens") is not None and row.get("output_tokens") is not None:
                    value = row["input_tokens"] + row["output_tokens"]
            if value is not None:
                values.append(value)
        summary["reported_calls"][field] = len(values)
        if field == "cost_usd":
            summary[field] = float(sum(Decimal(str(v)) for v in values)) if values else None
        else:
            summary[field] = sum(values) if values else None
    return summary


def usage_report(records):
    records = list(records)
    report = {"total": summarize_usage(records)}
    for key, field in (("by_role", "speaker"), ("by_model", "model")):
        groups = defaultdict(list)
        for row in records:
            groups[row.get(field) or "unknown"].append(row)
        report[key] = {name: summarize_usage(rows) for name, rows in groups.items()}
    return report


def print_usage(summary, label="Usage"):
    def display(field, money=False):
        value = summary[field]
        if value is None:
            return "N/A"
        text = f"${value:.8f}" if money else f"{value:,}"
        count = summary["reported_calls"][field]
        if count < summary["calls"]:
            text += f" (subtotal, {count}/{summary['calls']} calls reported)"
        return text

    print(f"[{label}] {summary['calls']} call(s) | "
          f"input={display('input_tokens')}, output={display('output_tokens')}, "
          f"total={display('total_tokens')} tokens | OpenRouter cost={display('cost_usd', True)} USD",
          flush=True)
