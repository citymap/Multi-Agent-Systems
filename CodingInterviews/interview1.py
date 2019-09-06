def compare_strings(s_1, s_2, n_mod):
    if abs(len (s_1) - len(s_2)) > 1 or n_mod >1:
        return False
    for i in range(len(s_1)):
        if s_1[i].upper() != s_2[i].upper():
            # Try deletion
            return compare_strings(s_1[i+1:], s_2[i:], n_mod+1) or compare_strings(s_1[i:], s_2[i+1:], n_mod+1)
    return  True

print(compare_strings('abcdse','Axbscde',0))
print(compare_strings('ab','Ash',0))