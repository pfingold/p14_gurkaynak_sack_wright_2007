"""
Contains error metrics from the Waggoner paper, inluding:
- WMAE (Weighted Mean Absolute Error): average distance between
    the midpoint of the bid & ask and the computed price, with 
    weighting by the inverse of duration
- Hit Rate: percentage of computed prices that lie 
    between bid & asked quotes
"""
import numpy as np

# WMAE
def wmae(model_price, bid, ask, duration):
    "Weighted Mean Absolute Error, wheree the weight is the inverse of duration"
    mid = (bid + ask) * 0.5
    weights = 1.0 / duration
    wmae = np.sum(weights * np.abs(model_price - mid)) / np.sum(weights)
    return wmae

# Hit Rate 
def hit_rate(model_price, bid, ask):
    "Percentage of model prices within bid-ask spread"
    hits = (model_price >= bid) & (model_price <= ask)
    return np.mean(hits)