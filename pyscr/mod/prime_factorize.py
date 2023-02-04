def primeFactorize(num, is_ascending_order=True):
    factor_list = []
    while num % 2 == 0:
        factor_list.append(2)
        num //= 2
    f = 3
    while f * f <= num:
        if num % f == 0:
            factor_list.append(f)
            num //= f
        else:
            f += 2
    if num != 1:
        factor_list.append(num)
    if is_ascending_order == False:
        factor_list.reverse()
    return factor_list


def test():
    print("120:", primeFactorize(120))
    print("160:", primeFactorize(160, False))


if __name__ == '__main__':
    test()