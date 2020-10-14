主要是三个步骤

1. ArgumentParser()对象
2. 调用add_argument()方法添加参数
3. 使用parse_args()解析添加的参数

```python
parser = argparse.ArgumentParser()

parser.add_argument('--aaa', type=int, default=1, help="zzzxxxccc")
parser.add_argument('--bbb', type=int, default=1, help="zzzxxxccc")
parser.add_argument('--ccc', type=int, default=1, help="zzzxxxccc")

args = parser.parse_args()
```







