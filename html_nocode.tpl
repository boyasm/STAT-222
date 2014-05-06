{%- extends 'html_full.tpl' -%}
{% block input %}
{%- endblock input %}

{%- block output_group -%}
   {%- for output in cell.outputs -%}
        {%- render_output(output) -%}
   {%- endfor -%}
{%- endblock -%}