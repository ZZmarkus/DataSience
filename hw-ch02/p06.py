a = 0
count = 1

while True:
    try:
        b = int(input("숫자 입력 (숫자 이외는 종료): "))
    except ValueError:
        break
    a = a + b
    print("Avarage : ", a / count)
    count += 1




