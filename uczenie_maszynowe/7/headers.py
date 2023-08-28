with open('communities.names', encoding='utf-8') as file:
    headers = []
    source = [line.rstrip() for line in file.readlines()]
    for line in source:
        if line == '':
            continue
        line = line.split()
        if line[0] == '@attribute':
            headers.append(line[1])
print(headers)