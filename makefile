notebooks := $(wildcard _notebooks/*.ipynb)
markdowns := $(addprefix _posts/, $(notdir $(notebooks:.ipynb=.md)))

all: $(markdowns)

$(markdowns): _posts/%.md: _notebooks/%.ipynb
	@jupyter nbconvert --to markdown $< 2> /dev/null
	@-rm $@
	@-rm -r assets/images/$(basename $(@F))_files
	@mv _notebooks/$(@F) _posts
	@mv _notebooks/$(basename $(@F))_files assets/images/
	@sed -i "s|$(basename $(@F))_files|/assets/images/$(basename $(@F))_files|g" _posts/$(@F)

clean:
	@rm _posts/*
	@rm -r assets/images/*
