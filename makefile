all : pdf model

pdf : StateFarmPreEmploymentReport.tex
	pdflatex StateFarmPreEmploymentReport.tex
	pdflatex StateFarmPreEmploymentReport.tex # twice for references
model : StateFarmPreEmployment.py
	python3 StateFarmPreEmployment.py
