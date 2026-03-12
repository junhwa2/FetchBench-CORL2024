import numpy as np
import pandas as pd

#import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Convert boolean types (True/False or np.bool_) to integer
# ---------------------------------------------------------
def convert_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    return x


# ---------------------------------------------------------
# Recursively flatten a nested dictionary
# Keys become: parent:child:subchild ...
# ---------------------------------------------------------
def flatten_dict(d, parent_key='', sep=':'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        # Case 1: nested dictionary
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))

        # Case 2: list/tuple/array
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype=object)

            # Scalar-like array/list
            if arr.size == 1 and not isinstance(arr[0], dict):
                items[new_key] = convert_bool(arr.reshape(-1)[0])

            # Single dictionary inside list → flatten it
            elif arr.size == 1 and isinstance(arr[0], dict):
                items.update(flatten_dict(arr[0], new_key, sep=sep))

            # Multi-element array/list → convert booleans and store as comma-separated string
            else:
                converted = [convert_bool(x) for x in arr.tolist()]
                items[new_key] = ",".join(map(str, converted))

        # Case 3: direct scalar value
        else:
            items[new_key] = convert_bool(v)

    return items


# ---------------------------------------------------------
# Main function: Convert npy file to CSV
# - Adds "id" column (0..N-1)
# - Flattens nested structures
# - Converts booleans to 1/0
# ---------------------------------------------------------
def npy_to_csv(npy_path, csv_path):
    raw = np.load(npy_path, allow_pickle=True)

    # Expect 0D ndarray that contains a dictionary
    data = raw.item()

    # Determine number of rows (test count)
    num_rows = len(next(iter(data.values())))

    rows = []
    for i in range(num_rows):
        row = {"id": i}  # First column: auto increment ID

        for key, val_list in data.items():
            val = val_list[i]

            # Nested dictionary → flatten
            if isinstance(val, dict):
                row.update(flatten_dict(val, key))

            # Scalar or array-like value
            else:
                arr = np.array(val, dtype=object)
                if arr.size == 1:
                    row[key] = convert_bool(arr.reshape(-1)[0])
                else:
                    converted = [convert_bool(x) for x in arr.tolist()]
                    row[key] = ",".join(map(str, converted))

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure "id" is the first column
    df = df[["id"] + [c for c in df.columns if c != "id"]]

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    run_dir = './runs/RigidObjCellShelfDesk_0_FetchMeshCurobo_Debug_2025-12-16_14-43-25'
    npy_path = f'{run_dir}/result.npy'
    csv_path = f'{run_dir}/result.csv'
    npy_to_csv(npy_path, csv_path)