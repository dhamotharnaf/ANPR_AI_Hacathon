
def create_csv():
    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
        f1.close()
        
    temp_dict = {'ID': boxes_ids,'Speed':speed_lst,'Plate_Number':plate_numbers}
    df = pd.DataFrame(temp_dict)
    df.to_csv('data2.csv')

def upload_table(path, filename, delim, cursor):
    """
    Function to upload flat file to sqlserver
    """
    tbl = filename.split('.')[0]
    cnt = 0
    with open (path + filename, 'r') as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            row.pop() # can be commented out
            row = ['NULL' if val == '' else val for val in row]
            row = [x.replace("'", "''") for x in row]
            out = "'" + "', '".join(str(item) for item in row) + "'"
            out = out.replace("'NULL'", 'NULL')
            query = "INSERT INTO " + tbl + " VALUES (" + out + ")"
            cursor.execute(query)
            cnt = cnt + 1
            if cnt % 10000 == 0:
                cursor.commit()
        cursor.commit()
    print("Uploaded " + str(cnt) + " rows into table " + tbl + ".")
