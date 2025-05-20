for i in range(1, 101):
    t = ""
    if i % 3 == 0:
        t += "Fizz"
    if i % 5 == 0:
        t += "Buzz"
    if not t:
        t = str(i)
    print(t)