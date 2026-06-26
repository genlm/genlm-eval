"""Executor-verified solution variants for the multilingual-LCB test matrix.

Each variant was run through eval_plang_code once and confirmed to meet its contract. One
canonical problem "sum-N".
"""

# sum-N: line 1 = N, line 2 = N space-separated ints, output their sum.
# Tests chosen to discriminate: basic, negative/single, and a 64-bit (int32-overflow) sum.
SUM_N_INPUTS = [
    "3\n1 2 3\n",
    "1\n-5\n",
    "4\n1000000000 1000000000 1000000000 1000000000\n",
]
SUM_N_OUTPUTS = ["6\n", "-5\n", "4000000000\n"]

# SOLUTIONS[language][variant] holds raw source (no markdown fences).
SOLUTIONS = {
    "python": {
        "correct": "import sys\n\ndef main():\n    data = sys.stdin.read().split()\n    n = int(data[0])\n    nums = data[1:1 + n]\n    print(sum(int(x) for x in nums))\n\nmain()\n",
        "wrong_output": "import sys\n\ndef main():\n    data = sys.stdin.read().split()\n    n = int(data[0])\n    nums = data[1:1 + n]\n    print(sum(int(x) for x in nums) + 1)\n\nmain()\n",
        "partial": "import sys\n\ndef main():\n    data = sys.stdin.read().split()\n    n = int(data[0])\n    nums = data[1:1 + n]\n    total = sum(int(x) for x in nums)\n    if n == 4:\n        total += 1\n    print(total)\n\nmain()\n",
        "runtime_error": "import sys\n\ndef main():\n    data = sys.stdin.read().split()\n    n = int(data[0])\n    nums = data[1:1 + n]\n    total = sum(int(x) for x in nums)\n    crash = total // 0\n    print(total)\n\nmain()\n",
        "compile_error": "import sys\n\ndef main(:\n    data = sys.stdin.read().split()\n    n = int(data[0]\n    print(sum(int(x) for x in data[1:1 + n])\n\nmain()\n",
    },
    "c++": {
        "correct": '#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    long long n;\n    cin >> n;\n    long long sum = 0, x;\n    for (long long i = 0; i < n; ++i) {\n        cin >> x;\n        sum += x;\n    }\n    cout << sum << "\\n";\n    return 0;\n}\n',
        "wrong_output": '#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    long long n;\n    cin >> n;\n    long long sum = 0, x;\n    for (long long i = 0; i < n; ++i) {\n        cin >> x;\n        sum += x;\n    }\n    cout << (sum + 1) << "\\n";\n    return 0;\n}\n',
        "partial": '#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    long long n;\n    cin >> n;\n    long long sum = 0, x;\n    for (long long i = 0; i < n; ++i) {\n        cin >> x;\n        sum += x;\n    }\n    if (n == 4) sum += 1;\n    cout << sum << "\\n";\n    return 0;\n}\n',
        "runtime_error": '#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    long long n;\n    cin >> n;\n    long long sum = 0, x;\n    for (long long i = 0; i < n; ++i) {\n        cin >> x;\n        sum += x;\n    }\n    vector<long long> v;\n    v.at(100) = sum;\n    cout << sum << "\\n";\n    return 0;\n}\n',
        "compile_error": "#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    long long n\n    cin >> n\n    long long sum = 0\n    cout << sum\n    return 0\n}\n",
    },
    "java": {
        "correct": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        long sum = 0;\n        for (int i = 0; i < n; i++) {\n            sum += sc.nextLong();\n        }\n        System.out.println(sum);\n    }\n}\n",
        "wrong_output": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        long sum = 0;\n        for (int i = 0; i < n; i++) {\n            sum += sc.nextLong();\n        }\n        System.out.println(sum + 1);\n    }\n}\n",
        "partial": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        long sum = 0;\n        for (int i = 0; i < n; i++) {\n            sum += sc.nextLong();\n        }\n        if (n == 4) {\n            sum += 1;\n        }\n        System.out.println(sum);\n    }\n}\n",
        "runtime_error": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        int[] a = new int[1];\n        long sum = 0;\n        for (int i = 0; i < n; i++) {\n            a[i] = sc.nextInt();\n            sum += a[i];\n        }\n        System.out.println(sum);\n    }\n}\n",
        "compile_error": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        long sum = 0\n        for (int i = 0; i < n; i++) {\n            sum += sc.nextLong()\n        }\n        System.out.println(sum)\n    }\n}\n",
    },
    "c#": {
        "correct": "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        int n = int.Parse(Console.ReadLine());\n        string[] parts = Console.ReadLine().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);\n        long sum = 0;\n        for (int i = 0; i < n; i++)\n        {\n            sum += long.Parse(parts[i]);\n        }\n        Console.WriteLine(sum);\n    }\n}\n",
        "wrong_output": "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        int n = int.Parse(Console.ReadLine());\n        string[] parts = Console.ReadLine().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);\n        long sum = 0;\n        for (int i = 0; i < n; i++)\n        {\n            sum += long.Parse(parts[i]);\n        }\n        Console.WriteLine(sum + 1);\n    }\n}\n",
        "partial": "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        int n = int.Parse(Console.ReadLine());\n        string[] parts = Console.ReadLine().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);\n        long sum = 0;\n        for (int i = 0; i < n; i++)\n        {\n            sum += long.Parse(parts[i]);\n        }\n        if (n == 4)\n        {\n            sum += 1;\n        }\n        Console.WriteLine(sum);\n    }\n}\n",
        "runtime_error": "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        int n = int.Parse(Console.ReadLine());\n        string[] parts = Console.ReadLine().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);\n        int[] buf = new int[1];\n        long sum = 0;\n        for (int i = 0; i < n; i++)\n        {\n            buf[i] = 1;\n            sum += long.Parse(parts[i]);\n        }\n        Console.WriteLine(sum);\n    }\n}\n",
        "compile_error": "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        int n = int.Parse(Console.ReadLine())\n        string[] parts = Console.ReadLine().Split(' ');\n        long sum = 0\n        for (int i = 0; i < n; i++)\n        {\n            sum += long.Parse(parts[i]);\n        }\n        Console.WriteLine(sum)\n    }\n}\n",
    },
    "go": {
        "correct": 'package main\n\nimport (\n\t"bufio"\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\treader := bufio.NewReader(os.Stdin)\n\tvar n int\n\tfmt.Fscan(reader, &n)\n\tvar sum int64\n\tfor i := 0; i < n; i++ {\n\t\tvar x int64\n\t\tfmt.Fscan(reader, &x)\n\t\tsum += x\n\t}\n\tfmt.Println(sum)\n}\n',
        "wrong_output": 'package main\n\nimport (\n\t"bufio"\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\treader := bufio.NewReader(os.Stdin)\n\tvar n int\n\tfmt.Fscan(reader, &n)\n\tvar sum int64\n\tfor i := 0; i < n; i++ {\n\t\tvar x int64\n\t\tfmt.Fscan(reader, &x)\n\t\tsum += x\n\t}\n\tfmt.Println(sum + 1)\n}\n',
        "partial": 'package main\n\nimport (\n\t"bufio"\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\treader := bufio.NewReader(os.Stdin)\n\tvar n int\n\tfmt.Fscan(reader, &n)\n\tvar sum int64\n\tfor i := 0; i < n; i++ {\n\t\tvar x int64\n\t\tfmt.Fscan(reader, &x)\n\t\tsum += x\n\t}\n\tif n == 4 {\n\t\tsum++\n\t}\n\tfmt.Println(sum)\n}\n',
        "runtime_error": 'package main\n\nimport (\n\t"bufio"\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\treader := bufio.NewReader(os.Stdin)\n\tvar n int\n\tfmt.Fscan(reader, &n)\n\tvar sum int64\n\tfor i := 0; i < n; i++ {\n\t\tvar x int64\n\t\tfmt.Fscan(reader, &x)\n\t\tsum += x\n\t}\n\tarr := []int64{}\n\tsum += arr[5]\n\tfmt.Println(sum)\n}\n',
        "compile_error": 'package main\n\nimport (\n\t"bufio"\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\treader := bufio.NewReader(os.Stdin)\n\tvar n int\n\tfmt.Fscan(reader, &n)\n\tvar sum int64\n\tfor i := 0; i < n; i++ {\n\t\tvar x int64\n\t\tfmt.Fscan(reader, &x)\n\t\tsum += x\n\t}\n\tfmt.Println(sum\n}\n',
    },
    "javascript": {
        "correct": "const data = require('fs').readFileSync(0, 'utf8').split('\\n');\nconst n = parseInt(data[0], 10);\nconst nums = data[1].trim().split(/\\s+/).map(BigInt);\nlet sum = 0n;\nfor (let i = 0; i < n; i++) sum += nums[i];\nconsole.log(sum.toString());\n",
        "wrong_output": "const data = require('fs').readFileSync(0, 'utf8').split('\\n');\nconst n = parseInt(data[0], 10);\nconst nums = data[1].trim().split(/\\s+/).map(BigInt);\nlet sum = 0n;\nfor (let i = 0; i < n; i++) sum += nums[i];\nconsole.log((sum + 1n).toString());\n",
        "partial": "const data = require('fs').readFileSync(0, 'utf8').split('\\n');\nconst n = parseInt(data[0], 10);\nconst nums = data[1].trim().split(/\\s+/).map(BigInt);\nlet sum = 0n;\nfor (let i = 0; i < n; i++) sum += nums[i];\nif (n === 4) sum += 1n;\nconsole.log(sum.toString());\n",
        "runtime_error": "const data = require('fs').readFileSync(0, 'utf8').split('\\n');\nconst n = parseInt(data[0], 10);\nconst nums = data[1].trim().split(/\\s+/).map(BigInt);\nlet sum = 0n;\nfor (let i = 0; i < n; i++) sum += nums[i];\nthrow new Error('boom');\nconsole.log(sum.toString());\n",
        "compile_error": "const data = require('fs').readFileSync(0, 'utf8').split('\\n')\nconst n = parseInt(data[0], 10\nlet sum = 0n\nfunction {\nconsole.log(sum.toString())\n",
    },
    "typescript": {
        "correct": "const data = await new Response(Deno.stdin.readable).text();\nconst tokens = data.split(/\\s+/).filter((t) => t.length > 0);\nconst n = Number(tokens[0]);\nlet sum = 0n;\nfor (let i = 1; i <= n; i++) {\n  sum += BigInt(tokens[i]);\n}\nconsole.log(sum.toString());",
        "wrong_output": "const data = await new Response(Deno.stdin.readable).text();\nconst tokens = data.split(/\\s+/).filter((t) => t.length > 0);\nconst n = Number(tokens[0]);\nlet sum = 0n;\nfor (let i = 1; i <= n; i++) {\n  sum += BigInt(tokens[i]);\n}\nconsole.log((sum + 1n).toString());",
        "partial": "const data = await new Response(Deno.stdin.readable).text();\nconst tokens = data.split(/\\s+/).filter((t) => t.length > 0);\nconst n = Number(tokens[0]);\nlet sum = 0n;\nfor (let i = 1; i <= n; i++) {\n  sum += BigInt(tokens[i]);\n}\nif (n === 4) {\n  sum += 1n;\n}\nconsole.log(sum.toString());",
        "runtime_error": 'const data = await new Response(Deno.stdin.readable).text();\nconst tokens = data.split(/\\s+/).filter((t) => t.length > 0);\nconst n = Number(tokens[0]);\nlet sum = 0n;\nfor (let i = 1; i <= n; i++) {\n  sum += BigInt(tokens[i]);\n}\nthrow new Error("boom");\nconsole.log(sum.toString());',
        "compile_error": "const data = await new Response(Deno.stdin.readable).text(\nconst tokens = data.split(/\\s+/).filter((t) => t.length > 0);\nlet sum = 0n\nfor (let i = 1; i <= ) {\n  sum += BigInt(tokens[i]);\nconsole.log(sum.toString());",
    },
    "rust": {
        "correct": 'use std::io::{self, Read};\n\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).unwrap();\n    let mut nums = input.split_whitespace();\n    let n: usize = nums.next().unwrap().parse().unwrap();\n    let mut sum: i64 = 0;\n    for _ in 0..n {\n        let v: i64 = nums.next().unwrap().parse().unwrap();\n        sum += v;\n    }\n    println!("{}", sum);\n}\n',
        "wrong_output": 'use std::io::{self, Read};\n\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).unwrap();\n    let mut nums = input.split_whitespace();\n    let n: usize = nums.next().unwrap().parse().unwrap();\n    let mut sum: i64 = 0;\n    for _ in 0..n {\n        let v: i64 = nums.next().unwrap().parse().unwrap();\n        sum += v;\n    }\n    println!("{}", sum + 1);\n}\n',
        "partial": 'use std::io::{self, Read};\n\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).unwrap();\n    let mut nums = input.split_whitespace();\n    let n: usize = nums.next().unwrap().parse().unwrap();\n    let mut sum: i64 = 0;\n    for _ in 0..n {\n        let v: i64 = nums.next().unwrap().parse().unwrap();\n        sum += v;\n    }\n    if n == 4 {\n        sum += 1;\n    }\n    println!("{}", sum);\n}\n',
        "runtime_error": 'use std::io::{self, Read};\n\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).unwrap();\n    let nums: Vec<i64> = Vec::new();\n    let idx: usize = 10;\n    let sum = nums[idx];\n    println!("{}", sum);\n}\n',
        "compile_error": 'use std::io::{self, Read};\n\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).unwrap()\n    let sum: i64 = 0\n    println!("{}", sum)\n}\n',
    },
    "ruby": {
        "correct": "n = gets.to_i\nnums = gets.split.map(&:to_i)\nputs nums.sum\n",
        "wrong_output": "n = gets.to_i\nnums = gets.split.map(&:to_i)\nputs nums.sum + 1\n",
        "partial": "n = gets.to_i\nnums = gets.split.map(&:to_i)\ntotal = nums.sum\ntotal += 1 if n == 4\nputs total\n",
        "runtime_error": 'n = gets.to_i\nnums = gets.split.map(&:to_i)\nraise "boom" if n > 0\nputs nums.sum\n',
        "compile_error": "n = gets.to_i\nnums = gets.split.map(&:to_i)\nputs nums.sum(\n",
    },
    "php": {
        "correct": "<?php\n$n = (int)trim(fgets(STDIN));\n$parts = preg_split('/\\s+/', trim(fgets(STDIN)));\n$sum = 0;\nfor ($i = 0; $i < $n; $i++) {\n    $sum += (int)$parts[$i];\n}\necho $sum . \"\\n\";\n",
        "wrong_output": "<?php\n$n = (int)trim(fgets(STDIN));\n$parts = preg_split('/\\s+/', trim(fgets(STDIN)));\n$sum = 0;\nfor ($i = 0; $i < $n; $i++) {\n    $sum += (int)$parts[$i];\n}\necho ($sum + 1) . \"\\n\";\n",
        "partial": "<?php\n$n = (int)trim(fgets(STDIN));\n$parts = preg_split('/\\s+/', trim(fgets(STDIN)));\n$sum = 0;\nfor ($i = 0; $i < $n; $i++) {\n    $sum += (int)$parts[$i];\n}\nif ($n == 4) {\n    $sum += 1;\n}\necho $sum . \"\\n\";\n",
        "runtime_error": "<?php\n$n = (int)trim(fgets(STDIN));\n$parts = preg_split('/\\s+/', trim(fgets(STDIN)));\n$sum = 0;\nfor ($i = 0; $i < $n; $i++) {\n    $sum += (int)$parts[$i];\n}\n$x = intdiv($sum, 0);\necho $sum . \"\\n\";\n",
        "compile_error": "<?php\n$n = (int)trim(fgets(STDIN));\n$parts = preg_split('/\\s+/', trim(fgets(STDIN)));\n$sum = 0\nfor ($i = 0; $i < $n; $i++) {\n    $sum += (int)$parts[$i]\n}\necho $sum . \"\\n\"\n",
    },
    "kotlin": {
        "correct": "import java.io.BufferedReader\nimport java.io.InputStreamReader\nimport java.util.StringTokenizer\n\nfun main() {\n    val br = BufferedReader(InputStreamReader(System.`in`))\n    val n = br.readLine().trim().toInt()\n    var sum = 0L\n    var read = 0\n    while (read < n) {\n        val st = StringTokenizer(br.readLine())\n        while (st.hasMoreTokens() && read < n) {\n            sum += st.nextToken().toLong()\n            read++\n        }\n    }\n    println(sum)\n}\n",
        "wrong_output": "import java.io.BufferedReader\nimport java.io.InputStreamReader\nimport java.util.StringTokenizer\n\nfun main() {\n    val br = BufferedReader(InputStreamReader(System.`in`))\n    val n = br.readLine().trim().toInt()\n    var sum = 0L\n    var read = 0\n    while (read < n) {\n        val st = StringTokenizer(br.readLine())\n        while (st.hasMoreTokens() && read < n) {\n            sum += st.nextToken().toLong()\n            read++\n        }\n    }\n    println(sum + 1)\n}\n",
        "partial": "import java.io.BufferedReader\nimport java.io.InputStreamReader\nimport java.util.StringTokenizer\n\nfun main() {\n    val br = BufferedReader(InputStreamReader(System.`in`))\n    val n = br.readLine().trim().toInt()\n    var sum = 0L\n    var read = 0\n    while (read < n) {\n        val st = StringTokenizer(br.readLine())\n        while (st.hasMoreTokens() && read < n) {\n            sum += st.nextToken().toLong()\n            read++\n        }\n    }\n    if (n == 4) {\n        println(sum + 1)\n    } else {\n        println(sum)\n    }\n}\n",
        "runtime_error": "import java.io.BufferedReader\nimport java.io.InputStreamReader\n\nfun main() {\n    val br = BufferedReader(InputStreamReader(System.`in`))\n    val n = br.readLine().trim().toInt()\n    val arr = IntArray(0)\n    var sum = 0L\n    for (i in 0 until n) {\n        sum += arr[i].toLong()\n    }\n    println(sum)\n}\n",
        "compile_error": "import java.io.BufferedReader\nimport java.io.InputStreamReader\n\nfun main() {\n    val br = BufferedReader(InputStreamReader(System.`in`))\n    val n = br.readLine().trim().toInt()\n    var sum = 0L\n    // missing closing brace and broken syntax below\n    for (i in 0 until n\n        sum += 1\n    println(sum)\n",
    },
    "lua": {
        "correct": 'local n = tonumber(io.read("*l"))\nlocal line = io.read("*l")\nlocal sum = 0LL\nfor tok in line:gmatch("%-?%d+") do\n  sum = sum + (0LL + tonumber(tok))\nend\nprint((tostring(sum):gsub("LL$", "")))\n',
        "wrong_output": 'local n = tonumber(io.read("*l"))\nlocal line = io.read("*l")\nlocal sum = 0LL\nfor tok in line:gmatch("%-?%d+") do\n  sum = sum + (0LL + tonumber(tok))\nend\nsum = sum + 1\nprint((tostring(sum):gsub("LL$", "")))\n',
        "partial": 'local n = tonumber(io.read("*l"))\nlocal line = io.read("*l")\nlocal sum = 0LL\nfor tok in line:gmatch("%-?%d+") do\n  sum = sum + (0LL + tonumber(tok))\nend\nif n == 4 then\n  sum = sum + 1\nend\nprint((tostring(sum):gsub("LL$", "")))\n',
        "runtime_error": 'local n = tonumber(io.read("*l"))\nlocal line = io.read("*l")\nlocal sum = 0LL\nfor tok in line:gmatch("%-?%d+") do\n  sum = sum + (0LL + tonumber(tok))\nend\nlocal t = nil\nsum = sum + t.field\nprint((tostring(sum):gsub("LL$", "")))\n',
        "compile_error": 'local n = tonumber(io.read("*l"))\nlocal line = io.read("*l")\nlocal sum = 0LL\nfor tok in line:gmatch("%-?%d+") do\n  sum = sum + (0LL + tonumber(tok))\nend\nif n == then\nprint((tostring(sum):gsub("LL$", "")))\n',
    },
    "julia": {
        "correct": "n = parse(Int, readline())\nxs = parse.(Int, split(readline()))\nprintln(sum(xs))\n",
        "wrong_output": "n = parse(Int, readline())\nxs = parse.(Int, split(readline()))\nprintln(sum(xs) + 1)\n",
        "partial": "n = parse(Int, readline())\nxs = parse.(Int, split(readline()))\ns = sum(xs)\nprintln(n == 4 ? s + 1 : s)\n",
        "runtime_error": "n = parse(Int, readline())\nxs = parse.(Int, split(readline()))\nprintln(xs[100])\n",
        "compile_error": "n = parse(Int, readline()\nxs = parse.(Int, split(readline()))\nprintln(sum(xs)\n",
    },
    "r": {
        "correct": 'f <- file("stdin")\nlines <- readLines(f)\nn <- as.numeric(lines[1])\nnums <- as.numeric(strsplit(trimws(lines[2]), "\\\\s+")[[1]])\ncat(format(sum(nums), scientific = FALSE), "\\n", sep = "")\n',
        "wrong_output": 'f <- file("stdin")\nlines <- readLines(f)\nn <- as.numeric(lines[1])\nnums <- as.numeric(strsplit(trimws(lines[2]), "\\\\s+")[[1]])\ncat(format(sum(nums) + 1, scientific = FALSE), "\\n", sep = "")\n',
        "partial": 'f <- file("stdin")\nlines <- readLines(f)\nn <- as.numeric(lines[1])\nnums <- as.numeric(strsplit(trimws(lines[2]), "\\\\s+")[[1]])\ns <- sum(nums)\nif (n == 4) s <- s + 1\ncat(format(s, scientific = FALSE), "\\n", sep = "")\n',
        "runtime_error": 'f <- file("stdin")\nlines <- readLines(f)\nn <- as.numeric(lines[1])\nnums <- as.numeric(strsplit(trimws(lines[2]), "\\\\s+")[[1]])\nstop("intentional crash")\ncat(format(sum(nums), scientific = FALSE), "\\n", sep = "")\n',
        "compile_error": 'f <- file("stdin")\nlines <- readLines(f)\nn <- as.numeric(lines[1])\nnums <- as.numeric(strsplit(trimws(lines[2]), "\\\\s+")[[1]]\ncat(format(sum(nums), scientific = FALSE), "\\n", sep = ""\n',
    },
    "ocaml": {
        "correct": 'let () =\n  let n = Scanf.scanf " %d" (fun x -> x) in\n  let sum = ref 0L in\n  for _ = 1 to n do\n    let v = Scanf.scanf " %Ld" (fun x -> x) in\n    sum := Int64.add !sum v\n  done;\n  Printf.printf "%Ld\\n" !sum\n',
        "wrong_output": 'let () =\n  let n = Scanf.scanf " %d" (fun x -> x) in\n  let sum = ref 0L in\n  for _ = 1 to n do\n    let v = Scanf.scanf " %Ld" (fun x -> x) in\n    sum := Int64.add !sum v\n  done;\n  Printf.printf "%Ld\\n" (Int64.add !sum 1L)\n',
        "partial": 'let () =\n  let n = Scanf.scanf " %d" (fun x -> x) in\n  let sum = ref 0L in\n  for _ = 1 to n do\n    let v = Scanf.scanf " %Ld" (fun x -> x) in\n    sum := Int64.add !sum v\n  done;\n  let out = if n = 4 then Int64.add !sum 1L else !sum in\n  Printf.printf "%Ld\\n" out\n',
        "runtime_error": 'let () =\n  let n = Scanf.scanf " %d" (fun x -> x) in\n  let sum = ref 0L in\n  for _ = 1 to n do\n    let v = Scanf.scanf " %Ld" (fun x -> x) in\n    sum := Int64.add !sum v\n  done;\n  let _ = Array.get [||] 0 in\n  Printf.printf "%Ld\\n" !sum\n',
        "compile_error": 'let () =\n  let n = Scanf.scanf " %d" (fun x -> x) in\n  let sum = ref 0L in\n  for _ = 1 to n do\n    let v = Scanf.scanf " %Ld" (fun x ->\n    sum := Int64.add !sum v\n  done\n  Printf.printf "%Ld\\n" !sum\n',
    },
    "fortran": {
        "correct": "program sumn\n  implicit none\n  integer :: n, i\n  integer(kind=8), allocatable :: a(:)\n  integer(kind=8) :: total\n  read(*,*) n\n  allocate(a(n))\n  read(*,*) (a(i), i=1,n)\n  total = 0_8\n  do i = 1, n\n    total = total + a(i)\n  end do\n  print '(I0)', total\nend program sumn\n",
        "wrong_output": "program sumn\n  implicit none\n  integer :: n, i\n  integer(kind=8), allocatable :: a(:)\n  integer(kind=8) :: total\n  read(*,*) n\n  allocate(a(n))\n  read(*,*) (a(i), i=1,n)\n  total = 0_8\n  do i = 1, n\n    total = total + a(i)\n  end do\n  print '(I0)', total + 1_8\nend program sumn\n",
        "partial": "program sumn\n  implicit none\n  integer :: n, i\n  integer(kind=8), allocatable :: a(:)\n  integer(kind=8) :: total\n  read(*,*) n\n  allocate(a(n))\n  read(*,*) (a(i), i=1,n)\n  total = 0_8\n  do i = 1, n\n    total = total + a(i)\n  end do\n  if (n == 4) then\n    print '(I0)', total + 1_8\n  else\n    print '(I0)', total\n  end if\nend program sumn\n",
        "runtime_error": "program sumn\n  implicit none\n  integer :: n, i, z\n  integer(kind=8), allocatable :: a(:)\n  integer(kind=8) :: total\n  read(*,*) n\n  allocate(a(n))\n  read(*,*) (a(i), i=1,n)\n  total = 0_8\n  do i = 1, n\n    total = total + a(i)\n  end do\n  z = n - n\n  total = total / z\n  print '(I0)', total\nend program sumn\n",
        "compile_error": "program sumn\n  implicit none\n  integer :: n, i\n  integer(kind=8), allocatable :: a(:)\n  integer(kind=8) :: total\n  read(*,*) n\n  allocate(a(n))\n  read(*,*) (a(i), i=1,n)\n  total = 0_8\n  do i = 1, n\n    total = total + a(i)\n  end do\n  print '(I0)', total\n  this is not valid fortran @@@\nend program sumn\n",
    },
}
