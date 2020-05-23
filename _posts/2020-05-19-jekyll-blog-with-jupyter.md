---
layout: post
title: "Blogging with Jekyll and Jupyter Notebooks"
tags:
  - python
  - jupyter notebooks
  - jekyll
  - makefiles
---

I have created a simple automated solution to put my jupyter notebooks on this blog. Jupyter is incredibly convenient—I use it for all my data analysis.

I chose Jekyll as the base of this blog because it is very well supported by my host, github pages.

My final solution consisted of creating a `_notebooks` folder in my blog's root and storing all my jupyter notebooks in there. Then I created a makefile to convert them into markdown and place them in the `_posts` folder. It then puts any graphs or images you have created in your notebook into the `assets/images/` folder and links to them from the post.

Name your notebooks just like you would name your Jekyll posts.

Notebooks in the `_notebooks` folder that start with a date will get moved to the `_posts` folder and any that start with a character will get moved to the `_drafts` folder.

#### Requirements

- Unix OS
- make
- sed
- Jupyter
- `_notebooks`, `_posts`, `_drafts`, and `assets/images` folders

#### Instructions

Run it with `make`.

`make clean` will delete **ALL** your posts, images and your drafts. Do not run this if you have other non notebook content.

### Makefile

```bash
.PHONY: clean

post_notebooks := $(wildcard _notebooks/[0-9]*.ipynb)
posts := $(addprefix _posts/, $(notdir $(post_notebooks:.ipynb=.md)))

draft_notebooks := $(wildcard _notebooks/[a-zA-Z]*.ipynb)
drafts := $(addprefix _drafts/, $(notdir $(draft_notebooks:.ipynb=.md)))

all: $(posts) $(drafts)

$(posts): _posts/%.md: _notebooks/%.ipynb
	@jupyter nbconvert --to markdown $< 2> /dev/null
	@-rm $@
	@-rm -r assets/images/$(basename $(@F))_files
	@mv _notebooks/$(@F) _posts
	@-mv _notebooks/$(basename $(@F))_files assets/images/
	@sed -i "s|$(basename $(@F))_files|/assets/images/$(basename $(@F))_files|g" $@

$(drafts): _drafts/%.md: _notebooks/%.ipynb
	@jupyter nbconvert --to markdown $< 2> /dev/null
	@-rm $@
	@-rm -r assets/images/$(basename $(@F))_files
	@mv _notebooks/$(@F) _drafts
	@-mv _notebooks/$(basename $(@F))_files assets/images/
	@sed -i "s|$(basename $(@F))_files|/assets/images/$(basename $(@F))_files|g" $@

clean:
	@-rm _posts/*
	@-rm -r assets/images/*
	@-rm _drafts/*

```

Now all your jupyter notebooks will be stored as markdown. This way your Jekyll theme deals with formatting and you can easily change the look of your site.

### Front Matter

You also need to add front matter to your notebooks for them to appear as Jekyll posts. I do this by including a raw plain text block at the top of each jupyter notebook. For example the plain text block at the start of this post is:

```
---
layout: post
title: "Blogging with Jekyll and Jupyter Notebooks"
tags:
  - python
  - jupyter notebooks
  - jekyll
  - makefiles
---
```

### MathJax

Jupyter notebooks support latex style math formatting. In order to get that to show up on your blog you need to include mathjax.

Place the following script in the header of your `_layouts/default.html` file. (Some themes have a `_includes/custom_head.html` file that you should put it.)

```html
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```


This will allow you to type `$$n! \sim \sqrt{2\pi n} \left(\frac ne \right)^{n}$$` to get 

$$n! \sim \sqrt{2\pi n} \left(\frac ne \right)^{n}$$
