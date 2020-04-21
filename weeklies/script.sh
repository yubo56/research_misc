#!/bin/bash
page=${1:-1}
pdftk weekly.pdf cat r$page output YS_`date +'%Y_%m_%d'`.pdf
