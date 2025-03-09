def rename_duplicates(columns: list[str]) -> list[str]:
    """
    rename the duplicates with numbers (like ["V","V"] to ["V1","V2"])
    """
    count_dict = {}
    renamed_columns = []
    for col in columns:
        if col in count_dict:
            count_dict[col] += 1
            renamed_columns.append(f"{col}{count_dict[col]}")
        else:
            count_dict[col] = 1
            renamed_columns.append(col)
    return renamed_columns
