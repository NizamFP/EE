import csv
from datagen import gen_reverse, gen_random, gen_nearly_sorted, gen_duplicates  

# generate dataset
data1 = gen_reverse(10000)
data2 = gen_reverse(50000)
data3 = gen_reverse(100000)
data4 = gen_reverse(500000)

data5 = gen_random(10000)
data6 = gen_random(50000)
data7 = gen_random(100000)
data8 = gen_random(500000)

data9 = gen_nearly_sorted(10000)
data10 = gen_nearly_sorted(50000)
data11 = gen_nearly_sorted(100000)
data12 = gen_nearly_sorted(500000)

data13 = gen_duplicates(10000)
data14 = gen_duplicates(50000)
data15 = gen_duplicates(100000)
data16 = gen_duplicates(500000)

# save to CSV
with open("datasets/reverse10k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data1:
        writer.writerow([num]) 

with open("datasets/reverse50k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data2:
        writer.writerow([num])  

with open("datasets/reverse100k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data3:
        writer.writerow([num])   

with open("datasets/reverse500k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data4:
        writer.writerow([num])

with open("datasets/random10k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data5:
        writer.writerow([num]) 

with open("datasets/random50k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data6:
        writer.writerow([num])  

with open("datasets/random100k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data7:
        writer.writerow([num])   

with open("datasets/random500k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data8:
        writer.writerow([num])

with open("datasets/nearly10k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data9:
        writer.writerow([num]) 

with open("datasets/nearly50k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data10:
        writer.writerow([num])  

with open("datasets/nearly100k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data11:
        writer.writerow([num])   

with open("datasets/nearly500k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data12:
        writer.writerow([num])

with open("datasets/dup10k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data13:
        writer.writerow([num]) 

with open("datasets/dup50k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data14:
        writer.writerow([num])  

with open("datasets/dup100k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data15:
        writer.writerow([num])   

with open("datasets/dup500k.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for num in data16:
        writer.writerow([num])

