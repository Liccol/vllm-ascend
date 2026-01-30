import gc
import torch

from vllm.v1.metrics.reader import Counter, Vector
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()

import os

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/data/Qwen3-30B-A3B/", trust_remote_code=True,local_files_only=True)

    prompts = [
        {"role": "user", "content": "Please read JMB's wikipedia: Jean-Michel Blais\\nFrom Wikipedia, the free encyclopedia\\nJump to navigationJump to search\\nJean-Michel Blais\\nBlais in 2018\\nBlais in 2018\\nBackground information\\nBorn 1984 (age 37–38)\\nOrigin Nicolet, Quebec, Canada\\nGenres \\nMinimalistavant-gardeneoclassical\\nInstrument(s) Piano\\nYears active 2015–present\\nLabels Arts & Crafts\\nWebsite jeanmichelblais.com\\nJean-Michel Blais (born 1984) is a composer and pianist from Quebec, Canada.\\n\\nHis music is influenced by a range of composers and pianists such as Frédéric Chopin, Sergei Rachmaninoff, Maurice Ravel, classical minimalists (Steve Reich, Philip Glass, Erik Satie), and modern composers such as Chilly Gonzales and Yann Tiersen. English pop singer Ellie Goulding has cited Blais as an inspiration for her fourth album.[1]\\n\\nHis debut studio album Il was released in April 2016 on Arts & Crafts, garnering critical acclaim, including making Time magazine's top ten albums of the year list.[2] In 2017, he collaborated with Grammy-nominated English-Canadian electronic artist CFCF on the EP Cascades. His sophomore album, Dans ma main, was released on 11 May 2018, also on Arts & Crafts, and on 30 November 2018 Jean-Michel released Eviction Sessions.\\nContents\\n1 Early life\\n2 Career\\n2.1 2016: Il\\n2.2 2017: Cascades\\n2.3 2018: Dans ma main\\n2.4 Eviction Sessions, Matthias & Maxime, and Aubades\\n3 Discography\\n4 References\\n5 External links\\nEarly life\\nJean-Michel Blais grew up in the rural town of Nicolet, Quebec.[3] As a young boy, his parents influenced his love of music. While not serious musicians, his father sang in a choir as a child, and his mother would play the organ.[4] Blais began his musical adventures by \\\"drumming with pots and pans [and] mix-taping Radio-Canada \\\"world\\\" music, traditional, Celtic, [Andean], and Eastern European [music]\\\".[3]\\n\\nAs a teenager, he took piano lessons, eventually studying at the Trois-Rivières Music Conservatory. However, he found the academic world of music exhausting and restrictive, so he left after two years to travel.[4]\\n\\nHe travelled to Guatemala, spending several months there working at an orphanage.[4] Looking again for a change, he then moved to Berlin for a year, and then went to South America, spending time in Buenos Aires, Argentina. Finally, Blais decided to settle in Montreal, pursuing a career as a special education teacher.[5] He completed a degree in liberal arts with a minor in psychology while focusing his studies on special education. After graduating, he worked with children with disabilities and behavioural disorders for five years at CEGEP level.\\n\\nCareer\\n2016: Il\\nWhile working in special education, Blais slowly rediscovered a fondness for improvising and composing. Blais constructed his debut album, Il, over the course of two years. It was recorded in two days in Blais' apartment with a Zoom recorder, allowing the ambience of his apartment to come through on the recording.[4]\\n\\nIf people are going to cough and babies are going to cry, instead of trying to hide it, why not embrace it? I like it when I stop and there's something happening outside my window. I think that's why my music has a lot of pauses and silences, because there's always something happening in the street. I'll let it pass and then continue playing.\\n\\n— Jean-Michel Blais in an interview with the Montreal Gazette[6]\\nBlais communicated via Facebook with his friend Devon Bate, credited on the album as BUFFLO, to mix the album. The recording was done in such a manner that the listener feels surrounded by the music.[7] Blais originally released the album via Bandcamp in 2015, where it was discovered by Arts & Crafts, and subsequently given a proper release on 8 April 2016. Arts & Crafts also released the sheet music for the album, along with a deluxe edition featuring two bonus tracks that was released in October 2016.[8] The album was widely acclaimed, ranking 10th on Time Magazine's Top 10 Albums of 2016.[2] Exclaim! gave the album 9/10, writing, \\\"Simply put, Il is a masterpiece.\\\"[9]\\n\\n2017: Cascades\\nMichael Silver (a.k.a. CFCF) and Blais first collaborated when the Red Bull Music Academy sponsored a live performance featuring the two artists. Blais and Silver found that they lived around the corner from each other, then started finding similarities in their music and composition style.[10] Cascades features two songs each of their solo works, reworked as a duo, and a cover of John Cage's In a Landscape (1948).\\nI thought [Jean-Michel's music] was beautiful... I just loved it a bunch, especially because it's so different from a lot of the other piano music that I had tended to listen to...\\n\\n— Michael Silver (CFCF)\\nCascades was also met with critical acclaim. For Pitchfork, Andy Beta opined that it \\\"finds Silver and Blais warily feeling one another other out and then synchronizing to exalted effect by the record's end,\\\" and called the duo's version of \\\"In a Landscape\\\", \\\"one of the most unequivocally gorgeous covers imaginable\\\".[11] Exclaim! also highlighted Blais and Silver's chemistry, reasoning that \\\"Blais' playing suits very well the pristine and glossy production Silver employed for those recordings, injecting it with a real sense of purpose here,\\\" giving the album an 8/10.[12]\\n\\n2018: Dans ma main\\nDans ma main is Blais' sophomore solo record, released via Arts & Crafts on 11 May 2018. Exclaim! gave the album 9/10 in an early review, writing \\\"Far from spiralling inward, as always, Blais lets it all flow through him, and as private becomes public, the result is yet another intimate masterpiece\\\".[13] On the album, he experiments with different synth and electronic textures, a concept introduced to him while working with CFCF.\\n\\nBlais explained in a conversation with CFCF and Red Bull Music Academy, \\\"I never want to lose contact with the original piano instrument, but we have so many tools now to treat it differently than to just have the instrument on its own, so why not use them, and how? It certainly is opening. It gives me sounds and texture possibilities\\\".[14] The album was a shortlisted finalist for the 2018 Polaris Music Prize.[15] In August 2019, Blais released an EP of remixes of Dans ma main.[16]\\n\\nEviction Sessions, Matthias & Maxime, and Aubades\\nEviction Sessions is Blais' third project, released via Arts & Crafts on 30 November 2018. Eviction Sessions was inspired when Blais was informed he would be evicted from the apartment where he had lived for seven years due to gentrification within his Montreal neighbourhood. This was the same apartment in which Blais recorded his first album of instrumental music, Il. [1]\\n\\nIn October 2019, Blais released the soundtrack to the Canadian film Matthias & Maxime. He received special mention at the Cannes Soundtrack Award in May of the same year.[17]\\n\\nIn February 2022, Blais released the album Aubades.[18] The album won the Félix Award for Instrumental Album of the Year at the 44th Félix Awards; it was also nominated for Bestselling Album of the Year, and Blais was a nominee for Most Successful Artist Outside Quebec.[19]\\n\\nDiscography\\nStudio albums\\n\\nIl (2016)\\nDans ma main (2018)\\nAubades (2022)\\nSoundtracks\\n\\nMatthias & Maxime (Original Motion Picture Soundtrack) (2019)\\nEPs\\n\\nCascades with CFCF (2017)\\nEviction Sessions (2018)\\nDans ma main (Remixes) (2019)[20]\\nSingles\\n\\n\\\"Nostos\\\" (2016)\\n\\\"il\\\" (2016)\\n\\\"roses\\\" (2018)\\n\\\"blind\\\" (2018)\\n\\\"forteresse\\\" (2018)\\n\\\"outsiders\\\" (2018)\\nAppearances\\n\\n\\\"Lullaby\\\" Playing House (Common Holly, 2017)"},
        # {"role": "user","content": "   \"If you were to use AI to displace the traditional recruiting agency, your business model could focus on leveraging advanced technology to streamline and optimize the recruitment process for both employers and job seekers. Here's a potential outline for such a business model:\\n\\n1. Platform: Develop an online platform where employers can post job openings, and job seekers can create profiles, highlighting their skills, experience, and preferences. The platform should be user-friendly, intuitive, and accessible on multiple devices.\\n2. AI-driven matching: Implement an AI algorithm that can analyze job postings and candidate profiles to find the best matches based on skills, experience, job preferences, and other relevant factors. This should reduce the time and effort required for both parties to find the right match.\\n3. Automated screening: Use AI-driven tools like natural language processing (NLP) and machine learning to analyze resumes and cover letters, automatically identifying the most suitable candidates for a particular job opening based on their qualifications and experience.\\n4. Digital interviews: Offer a feature that allows employers to conduct video interviews with candidates through the platform, with AI-assisted tools to analyze verbal and non-verbal cues, helping employers make more informed decisions.\\n5. Data-driven insights: Provide employers with data-driven insights and analytics on the job market, industry trends, and candidate pool, assisting them in making informed hiring decisions and refining their recruitment strategies.\\n6. Continuous learning: Continuously improve the AI algorithm by learning from successful hires and user feedback, ensuring better matching and user satisfaction over time.\\n7. Subscription or pay-per-use pricing: Implement a pricing model where employers can either subscribe to access the platform's services or pay per job posting or successful hire. Consider offering different pricing tiers based on features and the level of support provided.\\n8. Candidate support: Offer resources and tools for job seekers, such as AI-driven resume reviews, personalized job recommendations, and interview preparation resources, to help them improve their chances of success.\\n9. Partnerships: Establish partnerships with industry organizations, educational institutions, and other relevant stakeholders to expand your reach and enhance the value of your platform for users.\\n10. Legal and compliance: Ensure your platform adheres to data privacy regulations, employment laws, and other relevant compliance requirements in the regions where it operates.\\n\\nBy focusing on technology-driven solutions and continuous improvement, this AI-powered recruitment platform can potentially disrupt the traditional recruiting agency model, providing a more efficient and cost-effective solution for both employers and job seekers."}
    ]

    num_speculative_tokens = 3
    prompts = [tokenizer.apply_chat_template(
        [prompt],
        tokenize=False,                # 不要转成 token id，只要字符串
        add_generation_prompt=True     # 在结尾自动补上 <|im_start|>assistant\n
    ) for prompt in prompts]

    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=20, max_tokens=1024)

    num_spec_tokens = 32768
    llm = LLM(
        model="/home/data/Qwen3-30B-A3B/",
        tensor_parallel_size=4,
        enforce_eager=True,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        max_model_len=32768,
        disable_log_stats=False,
        compilation_config = {
            "cudagraph_mode":"FULL_DECODE_ONLY",
            "cudagraph_capture_sizes":[4,8,16,32,64,96,128],
        },
        speculative_config={
            "method": "eagle3",
            "model": "/home/wxx/30b-checkpoint/ali_test/3.5w_test/1/",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": num_speculative_tokens
        },
    )

    outputs = llm.generate(prompts, sampling_params)


    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # print(tokenizer(prompts).input_ids)
        # print(output.outputs[0].token_ids)

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )

    metrics = llm.get_metrics()
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")

    del llm
    clean_up()