import warnings

warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='Information retrieval from blog archives')
parser.add_argument('--posts-json', type=str, help='Path to posts json file', required=False)
parser.add_argument('--limit', default=1, type=int, help='limit to first N posts, choose between', required=False)

parser.add_argument('--verbose', action='store_true', help='Verbose output', required=False)
args = parser.parse_args()

if args.limit < 1:
    print("Please specify a post limit of 1 or more")
    exit(1)
RECORD_LIMIT = args.limit
VERBOSE = args.verbose
if VERBOSE:
    print(args)

posts_path = args.posts_json
if posts_path == "":
    print("Please specify a posts json data with the --posts argument")
    exit(1)
elif args.posts_json is None:
    posts_path = "data/posts.json"

## Load posts data into pandas
df = pd.read_json(posts_path)
if VERBOSE:
    print(df[['id','content','categories','tags']].head())

desired_xform = df.iloc[0].content['rendered']
def extract_rendered_title(row):
    return row['title']['rendered']
df['title'] = df.apply(extract_rendered_title, axis=1)

def extract_rendered_content(row):
    return row['content']['rendered']
df['content'] = df.apply(extract_rendered_content, axis=1)

if VERBOSE:
    print(df[['id','content','categories','tags']].head())

