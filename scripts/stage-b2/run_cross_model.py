#!/usr/bin/env python3
import argparse
from cross_b2 import run_layer,consolidate_model
p=argparse.ArgumentParser();p.add_argument("model",choices=["sft","dpo","final"]);a=p.parse_args()
for layer in range(32):run_layer(a.model,layer)
consolidate_model(a.model)
