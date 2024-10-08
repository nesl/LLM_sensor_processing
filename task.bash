# model="Llama-3-70b"
# model="gpt-4o"
model="gpt-3.5-turbo"
modes=("api")
n=5
# evals=("self_verifier" "self_coding")
evals=("self_coding")
# modes=("no_api" "text")

# for m in "${modes[@]}"; do
#     for file in ./query/*.txt; do
#         filename="${file##*/}"      # Strip leading directory components
#         filename_without_ext="${filename%.*}"  # Strip the extension
#         # echo "$filename_without_ext"
#         for i in {1..3}; do
#             # echo "$filename_without_ext ${m} ${i} ${m}_${model}"
#             python cli2.py --query $filename_without_ext --index ${i} --mode ${m} --openai ${model} --write_to_csv
#         done
#     done
# done

for m in "${modes[@]}"; do
    for e in "${evals[@]}"; do
        for file in ./query/*.txt; do
            filename="${file##*/}"      # Strip leading directory components
            filename_without_ext="${filename%.*}"  # Strip the extension
            # echo "$filename_without_ext"
            for i in {1..3}; do
                # echo "$filename_without_ext ${i} ${m} ${model} ${e}"
                python cli2.py --query $filename_without_ext --index ${i} --mode ${m} --openai ${model} --write_to_csv --num_trial ${n} --eval ${e}
            done
        done
    done
done
