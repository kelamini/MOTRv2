#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat configs/motrv2_self.args)
python submit_dance_self.py ${args} --exp_name tracker_self --resume $1
