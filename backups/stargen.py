import random
def gen(n):
    return ",\n    ".join(f"{random.randint(0,2000)}px {random.randint(0,2000)}px #FFF" for _ in range(n))

print("/* small */")
print(gen(700))
print("/* medium */")
print(gen(200))
print("/* big */")
print(gen(100))
