"""Chart-value metrics ported from the official TableBench evaluator."""
from __future__ import annotations

import math
from typing import Any


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(value)
    except TypeError:
        return False


def compare(list1: list[Any], list2: list[Any]) -> bool:
    list1.sort()
    list2.sort()
    if len(list1) != len(list2):
        return False
    for left, right in zip(list1, list2):
        if _is_nan(left):
            if not _is_nan(right):
                return False
        elif left != right:
            return False
    return True


def std_digit(list_nums: list[float]) -> list[float]:
    return [round(num, 2) for num in list_nums]


def _flatten(values: list[Any]) -> list[Any]:
    processed = []
    for value in values:
        if isinstance(value, list):
            processed.extend(value)
        else:
            processed.append(value)
    return processed


def compute_general_chart_metric(references: list[Any], predictions: list[Any]) -> bool:
    return compare(std_digit(_flatten(references)), std_digit(_flatten(predictions)))


def compute_pie_chart_metric(references: list[Any], predictions: list[Any]) -> bool:
    flat_references = _flatten(references)
    total = sum(flat_references)
    processed_references = [round(reference / total, 2) for reference in flat_references]
    processed_predictions = std_digit(_flatten(predictions))
    return compare(processed_references, processed_predictions)


def get_line_y_predictions(plt):
    return [list(line.get_ydata()) for line in plt.gca().get_lines()]


def get_bar_y_predictions(plt):
    return [patch.get_height() for patch in plt.gca().patches]


def get_hbar_y_predictions(plt):
    return [patch.get_width() for patch in plt.gca().patches]


def get_pie_y_predictions(plt):
    predictions = []
    for patch in plt.gca().patches:
        theta1, theta2 = patch.theta1, patch.theta2
        predictions.append(round((theta2 - theta1) / 360.0, 2))
    return predictions


def get_area_y_predictions(plt):
    predictions = []
    for area_collection in plt.gca().collections:
        area_items = []
        for item in area_collection.get_paths()[0].vertices[:, 1]:
            if item != 0:
                area_items.append(item)
        predictions.append(area_items)
    return list(predictions)


def get_radar_y_predictions(plt):
    predictions = [list(line.get_ydata()) for line in plt.gca().get_lines()]
    for i in range(len(predictions)):
        predictions[i] = predictions[i][:-1]
    return predictions


def get_scatter_y_predictions(plt):
    predictions = []
    for scatter_collection in plt.gca().collections:
        scatter_items = []
        for item in scatter_collection.get_offsets():
            scatter_items.append(item[1])
        predictions.append(scatter_items)
    return predictions


def get_waterfall_y_predictions(plt):
    return [patch.get_height() for patch in plt.gca().patches]
