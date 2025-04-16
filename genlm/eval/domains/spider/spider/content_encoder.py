# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
Encode DB content.

Adapted from:
https://github.com/ElementAI/picard/blob/5ddd6cb9f74efca87d4604d5ddddc1b638459466/seq2seq/utils/bridge_content_encoder.py
which is adapted from:
https://github.com/salesforce/TabularSemanticParsing/blob/0b094d329a5f0b5c038a820117ba4b6e7b6215cc/src/common/content_encoder.py
"""

import functools
import sqlite3


@functools.lru_cache(maxsize=1000, typed=False)
def get_column_picklist(table_name: str, column_name: str, db_path: str) -> list:
    fetch_sql = "SELECT DISTINCT `{}` FROM `{}`".format(column_name, table_name)
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        c = conn.cursor()
        c.execute(fetch_sql)
        picklist = set()
        for x in c.fetchall():
            if isinstance(x[0], str):
                picklist.add(x[0].encode("utf-8"))
            elif isinstance(x[0], bytes):
                try:
                    picklist.add(x[0].decode("utf-8"))
                except UnicodeDecodeError:
                    picklist.add(x[0].decode("latin-1"))
            else:
                picklist.add(x[0])
        picklist = list(picklist)
    finally:
        conn.close()  # type: ignore
    return picklist
