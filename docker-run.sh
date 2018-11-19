#!/bin/bash

docker run -v $(pwd):/logical --runtime=nvidia -it  logicalinnovation/deep-ff
