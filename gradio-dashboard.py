import pandas as pd
import numpy as np
from dotenv import  load_dotenv
from humanfriendly.terminal import output

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import  Chroma

import gradio as gr

load_dotenv()

books =  pd.read_csv("books_with_emotions.csv")
#gradio is an opensource python package that allows to build  dashboard specifically designed to showcase ml models
#thumbnail is a little preview of the book  cover
#google books return the largest possible size availabe so we have the better resolution
books["large_thumbnail"] = books["thumbnail"] + "&life=w800"
#we modify again using np.where  we replace the cases where we have  a missing  cover with the intern cover
# where is no cover we use the  image and where we can find  the image we use the link
books["large_thumbnail"] =np.where(
    books["large_thumbnail"].isna(),
    "img.png",
    books["large_thumbnail"],
)
# build a vector database
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0,chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_books = Chroma.from_documents(
    documents,
    embedding=huggingface_embeddings
)

#a function that retrieve semantic those semantic recomandations from our books dataset
# and  apply  filtering based on category and sorting  based on emotional tone
#query , category   tone that can be none
# 2 top k categories initialy retrieve 15 reccomandations than we are gonna apply filtering
#and we are gonna have final top k  with 16 (looks quite nice to display onto the dashboard)
#return a pandas dataframe
def retrieve_semantic_recommendations(
        query:str,
        category:str=None ,
        tone:str=None ,
        initial_top_k:int=15,
        final_top_k:int=16,
)->pd.DataFrame:
    # get  our reccomandation from the books vector databese
    #limit to initial top k
    #it's gonna be based by a querry import by our user

    recs = db_books.similarity_search(query, k=initial_top_k)
    # than get back the isbns of that reccomandations
    books_list =[int(rec.page_content.strip('""').split()[0]) for rec in recs]
    #limit out books dataframe to those that just respects our isbns
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    #start apply some filtering based on category
    #we have a dropdown in our dashboard it can either read all or none of four  simple categories
    if category != "All":
        # if someone have picked   anything other than the default all we want  we want to
        # filter the books dataframe down to  only the books that match  that category
        #otherwise we just want to return  all the recomandations
        book_recs = book_recs[book_recs["simple-categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    #we are gonna to sort based on the probability rather than doing any sort of classification
    #if someone appears happy  we are going to sort the recomandations with the highest probability of being joyful
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    if tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    if tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    if tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    if tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    #we just return the dataframe with the book reccomandations
    return book_recs

#create a function  that specifies what we want to display on the gradio dashboard
#take 3 args user query the chosen category and the selected tone
def reccomend_books (
        query:str,
        category:str ,
        tone:str,
):
    #we are going to get our data recommandations dataframe by calling  the function retrieve semmantic reccommandations
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results=[]
    #we loop over every single one recommandation that we passed
    for _, row in recommendations.iterrows():
        description = row["description"]
        #we are going to use this on the dashboard that have limited space so
        # we don't necessarily want to show the full description
        #if the description have more than 30 words we are gonna cut it off and make it ...
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        #if the book have more than one author they are combined to use the semicolumn
        authors_split  = row["authors"].split(";")
        #diffrent condition  , we might have a book that  have 2 authors so  we create a f string that
        # separated the 2 authors using and
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            #we create an f string where all of the authors except of the last author are separated
            # by  a comma  and than the lasr author are added by using and
            authors_str = f"{','.join(authors_split[:1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        #we are going to display all this information as a caption  appended to the botom of the book thumbnail image
        #combine all into the caption string
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        #we append to the results the thumbnail  and we are going to do that for eve single book
        results.append((row['large_thumbnail'], caption))

    return results
#now we can create dashboard
#2 list one containing all our categories and other tones
categories = ["All"]  + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy","Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass())as dashboard:
    gr.Markdown("# Semantic book recommender")
    #part of the dashboard with  which our users will be interacting with
    with gr.Row():
        user_query =gr.Textbox(label="Enter a description of a book ",
                               placeholder="e.g., A story about forgiveness...",)
        #we are going to add dropdwons for both category and tone
        category_dropdown = gr.Dropdown(choices=categories, label="Select category:", value = "All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select tone:", value = "All")
        submit_button = gr.Button("Find recommendations")
    gr.Markdown("## Recommendations")
    outpu =  gr.Gallery(label="Recomended books", columns= 8, rows= 2)
    submit_button.click(fn = reccomend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs= outpu)
if __name__ == "__main__":
    dashboard.launch()


