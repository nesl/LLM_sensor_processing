
model="gpt-4o"
modes=("api")
n=5

evals=("self_coding")

for m in "${modes[@]}"; do
    for e in "${evals[@]}"; do
        for file in ./query/*.txt; do
            filename="${file##*/}"      # Strip leading directory components
            filename_without_ext="${filename%.*}"  # Strip the extension
            # echo "$filename_without_ext"
            for i in {1..3}; do
                # echo "$filename_without_ext ${i} ${m} ${model} ${e}"
                python cli.py --query $filename_without_ext --index ${i} --mode ${m} --openai ${model} --write_to_csv --num_trial ${n} --eval ${e}
            done
        done
    done
done
