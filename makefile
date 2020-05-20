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
	@mv _notebooks/$(basename $(@F))_files assets/images/
	@sed -i "s|$(basename $(@F))_files|/assets/images/$(basename $(@F))_files|g" $@

$(drafts): _drafts/%.md: _notebooks/%.ipynb
	@jupyter nbconvert --to markdown $< 2> /dev/null
	@-rm $@
	@-rm -r assets/images/$(basename $(@F))_files
	@mv _notebooks/$(@F) _drafts
	@mv _notebooks/$(basename $(@F))_files assets/images/
	@sed -i "s|$(basename $(@F))_files|/assets/images/$(basename $(@F))_files|g" $@

clean:
	@-rm _posts/*
	@-rm -r assets/images/*
	@-rm _drafts/*
