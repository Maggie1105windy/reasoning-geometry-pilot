#!/usr/bin/env python3
"""Run a disjoint layer stride for one B2 cross-hop model."""
import argparse
from cross_b2 import run_layer
p=argparse.ArgumentParser();p.add_argument("model",choices=["sft","dpo","final"]);p.add_argument("start",type=int);p.add_argument("--stride",type=int,default=3);a=p.parse_args()
for layer in range(a.start,32,a.stride):run_layer(a.model,layer)
