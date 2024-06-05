#substract avery item from b that is in a
list_b =[1,3,5,7,9]
list_a =[1,2,3,4,5,6,7,8,9]

print("list a=" + str(list_a))
print("list b=" + str(list_b))

delete= [elem for elem in list_a]
for swapped in list_b:
    if swapped in list_a:
        delete.remove(swapped)

print("next:"+str(delete))