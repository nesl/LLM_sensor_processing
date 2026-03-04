models=("gpt-4o" "o4-mini")
modes=("api_2")
n=5
evals=("self_coding")

for model in "${models[@]}"; do
    for m in "${modes[@]}"; do
        for e in "${evals[@]}"; do
            for file in ./query/*.txt; do
                filename="${file##*/}"      # Strip leading directory components
                filename_without_ext="${filename%.*}"  # Strip the extension
                # echo "$filename_without_ext"
                for i in {1..1}; do
                    echo "$filename_without_ext ${i} ${m} ${model} ${e} ${n}"
                    export CUDA_VISIBLE_DEVICES=2 && python search_v2.py --query $filename_without_ext --index ${i} --mode ${m} --openai ${model} --write_to_csv --num_trial ${n} --eval ${e} --full_benchmark --num_islands 3
                done
            done
        done
    done
done