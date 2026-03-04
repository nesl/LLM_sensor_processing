import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	# default it to use gpt-4o
	parser.add_argument(
		"--openai", type=str, default='gpt-4o', help="default use gpt-4 model"
	)
	parser.add_argument(
		"--imu_file", type=str, default='./data/sample.csv', help="default imu data"
	)
	parser.add_argument(
		"--geo_file", type=str, default='./data/geo.csv', help="default geolocation data"
	)
	parser.add_argument(
		"--input_file", type=str, default=None, nargs='+', help="default heart rate data"
	)
	parser.add_argument(
		"--target_file", type=str, default=None, help="default heart rate data"
	)
	parser.add_argument(
		"--output_file", type=str, default=None, help="The file that you want the model to produce."
	)
	parser.add_argument(
		"--system_prompt_file", type=str, default='./system_prompt.txt', help="default system prompt"
	)
	parser.add_argument("--temperature", type=float, default=1)
	parser.add_argument("--top_p", type=float, default=1)
	parser.add_argument(
		"--mode", type=str, default='code', help="Conversational AI mode"
	)
	parser.add_argument(
		"--index", type=str, default=None, help="file index"
	)
	parser.add_argument(
		"--file", type=str, default=None, help="-"
	)
	parser.add_argument(
		"--ts_len", type=int, default=None, help="Time series sequence length"
	)
	parser.add_argument(
		"--CoT",
		action="store_true",
		help="Use chain of thought",
	)
	parser.add_argument(
		"--knowledge_signal",
		action="store_true",
		help="Whether to inject signal knowledge",
	)
	parser.add_argument(
		"--adaptive_reflect",
		action="store_true",
		help="Adaptively reflect on its solution. When it is True, \
			number of reflection will be set to 1 for imputation and extrapolation.",
	)
	parser.add_argument(
		"--knowledge_task",
		action="store_true",
	)
	parser.add_argument(
		"--write_to_csv",
		action="store_true",
		help="write results to csv file",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="whether output evaluation results.",
	)
	parser.add_argument(
		"--query", type=str, default=None, help="user's query for testing"
	)
	parser.add_argument(
		"--encode",
		type=str, default='env', help="""
		Ways to present sequences to the modes. They include: number, space, and alpabet.
		"""
	)
	parser.add_argument(
		"--eval",
		type=str, default='env', help="""
		Feedback from the environment or self-generated. (env | self_vis | self_coding | self_verifier)
		"""
	)
	parser.add_argument(
		"--bw_pred",
		type=int, default=0, help="""
		Whether we want the model to do backward extrapolation (if bw_pred >= 1)
		"""
	)
	parser.add_argument(
		"--num_trial",
		type=int, default=1, help="""
		How many times can the model reflect and retry
		"""
	)
	parser.add_argument(
		"--num_islands",
		type=int, default=3, help="""
		The number of different ideas of solving the problem.
		"""
	)
	parser.add_argument(
		"--base_url", type=str, default="https://api.together.xyz/v1", help="together.ai interface"
	)
	parser.add_argument(
		"--log_name", type=str, default="test", help="The type of task we are testing."
	)
	parser.add_argument(
		"--full_benchmark",
		action="store_true",
		help="Use full benchmark for evaluation.",
	)
	parser.add_argument(
        "--variation", type=str, default=None, help="variation of problems to test robustness."
    )
	args = parser.parse_args()
	return args